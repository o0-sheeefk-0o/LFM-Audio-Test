"""発話の切り出し（fastrtc の ReplyOnPause の調整版）

fastrtc の既定の切り出しは 0.6 秒ごとの区切りで判定し、
  (1) 話し始めを検出した区切りより前を捨てる
  (2) 話し始めた後、発話が 0.1 秒未満の区切りが 1 つ来た時点で発話終了とみなす
ため、言い始めや、言葉の途中で間が空いた後ろが切れて、質問の冒頭（0.6〜1.2 秒）しかモデルに届かないことがあった。
自己紹介のあとに質問の冒頭だけを渡すと、何を聞いても自己紹介が返ってくる（同じ返答が繰り返される原因の 1 つ。
モデル自体にも前の返答を繰り返す癖があり、その対策は engine.py）。

また、応答の生成は再生より速く終わる（RTF 約 0.5）ので、既定では生成が終わった時点でマイク入力の受け付けが再開され、
ブラウザでまだ再生中の応答音声の残響や物音（-40〜-50 dBFS）を次の発話として拾っていた。
意味のない入力にもツール判定が反応して時刻ツールなどを呼び、同じ返答が繰り返される原因になっていた。

ここでは次のように変える。
  - 区切りを細かくし（0.4 秒）、直近 1.2 秒の窓で VAD をかけて最新の区切りに含まれる発話の長さを求める
  - 話し始める前の区切りも先頭に残す（言い始めを捨てない）
  - 無音が end_silence_s 続いたら発話終了（言葉の途中の短い間では切らない）
  - 応答音声の再生が終わるまで（＋余裕）はマイク入力を捨てる
  - 音量が min_level_db 未満の音では話し始めとみなさない
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from fastrtc import AlgoOptions, ReplyOnPause
from fastrtc.pause_detection.silero import SileroVadOptions
from fastrtc.reply_on_pause import AppState
from fastrtc.utils import create_message, split_output

logger = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16_000


@dataclass
class VadConfig:
    chunk_s: float = 0.4  # この長さごとに判定する
    window_s: float = 1.2  # VAD にかける窓の長さ（前後の文脈があるほうが判定が安定する）
    start_speech_s: float = 0.15  # 最新の区切りの発話がこれ以上なら話し始め
    silence_speech_s: float = 0.05  # 最新の区切りの発話がこれ未満なら無音
    end_silence_s: float = 0.8  # 話し始めた後、無音がこれだけ続いたら発話終了
    preroll_chunks: int = 2  # 話し始める前の区切りを何個残すか
    max_utterance_s: float = 30.0  # 雑音などで終わらないときの上限
    threshold: float = 0.5  # Silero VAD の発話確率のしきい値（マイクの音量が小さければ下げる）
    min_level_db: float = -35.0  # 話し始めの区切りの音量（20ms ごとの RMS の最大値, dBFS）がこれ未満なら無視する
    playback_guard_s: float = 0.5  # 応答音声の再生終了からマイク入力を受け付けるまでの余裕（ブラウザ側のバッファと残響）


# 画面から変更する（単一ユーザーの検証用アプリなのでプロセス全体で 1 つ）
VAD = VadConfig()


@dataclass
class UtteranceState(AppState):
    window: deque = field(default_factory=deque)  # 直近の区切り（VAD の文脈と、話し始め前の先頭用）
    silence_s: float = 0.0

    def new(self) -> "UtteranceState":
        return UtteranceState()


def level_db(audio: np.ndarray, sampling_rate: int) -> float:
    """20ms ごとの RMS の最大値（dBFS）"""
    x = audio.astype(np.float32)
    if np.issubdtype(audio.dtype, np.integer):
        x /= 32_768
    n = sampling_rate // 50
    frames = x[: len(x) // n * n].reshape(-1, n)
    if len(frames) == 0:
        return float("-inf")
    return float(20 * np.log10(np.sqrt((frames**2).mean(1)).max() + 1e-9))


class UtteranceReplyOnPause(ReplyOnPause):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("algo_options", AlgoOptions(audio_chunk_duration=VAD.chunk_s))
        super().__init__(*args, **kwargs)
        self.state = UtteranceState()
        # ブラウザで応答音声の再生が終わる見込みの時刻（time.monotonic()）
        self.playback_end = 0.0

    def reset(self) -> None:
        # 親クラスの reset() は AppState() を直接作るので、応答のたびに拡張した状態が消えてしまう
        super().reset()
        self.state = UtteranceState()

    def copy(self) -> "UtteranceReplyOnPause":
        # 親クラスの copy() は ReplyOnPause を直接作るので上書きする（接続ごとに呼ばれる）
        return UtteranceReplyOnPause(
            self.fn,
            startup_fn=self.startup_fn,
            algo_options=self.algo_options,
            model_options=self.model_options,
            can_interrupt=self.can_interrupt,
            expected_layout=self.expected_layout,
            output_sample_rate=self.output_sample_rate,
            output_frame_size=self.output_frame_size,
            input_sample_rate=self.input_sample_rate,
            model=self.model,
            needs_args=self.needs_args,
        )

    def emit(self):
        # 出力音声は生成でき次第キューに積まれ、実時間で再生される。積んだ長さから再生の終了時刻を見積もる
        output = super().emit()
        audio, _ = split_output(output)
        if audio is not None:
            sampling_rate, array = audio[0], audio[1]
            self.playback_end = max(self.playback_end, time.monotonic()) + array.shape[-1] / sampling_rate
        return output

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        # 応答音声の再生中はマイクがその声（残響）を拾うので捨てる
        if time.monotonic() < self.playback_end + VAD.playback_guard_s:
            return
        super().receive(frame)

    def _speech_in_last_chunk(self, window: np.ndarray, chunk_len: int, sampling_rate: int) -> float:
        options = SileroVadOptions(threshold=VAD.threshold, min_speech_duration_ms=100, speech_pad_ms=30, min_silence_duration_ms=100)
        _, chunks = self.model.vad((sampling_rate, window), options)
        # タイムスタンプは 16kHz に変換後のサンプル位置。窓の末尾 chunk_len ぶんと重なる発話の長さを数える
        total = len(window) * VAD_SAMPLE_RATE // sampling_rate
        start = total - chunk_len * VAD_SAMPLE_RATE // sampling_rate
        overlap = sum(max(0, min(c["end"], total) - max(c["start"], start)) for c in chunks)
        return overlap / VAD_SAMPLE_RATE

    def determine_pause(self, audio: np.ndarray, sampling_rate: int, state: UtteranceState) -> bool:  # type: ignore[override]
        if len(audio) / sampling_rate < VAD.chunk_s:
            return False
        state.buffer = None

        state.window.append(audio)
        while sum(len(a) for a in state.window) / sampling_rate > VAD.window_s + 1e-6 and len(state.window) > 1:
            state.window.popleft()
        speech_s = self._speech_in_last_chunk(np.concatenate(state.window), len(audio), sampling_rate)

        if not state.started_talking:
            if speech_s >= VAD.start_speech_s:
                level = level_db(audio, sampling_rate)
                if level < VAD.min_level_db:
                    logger.info("ignored quiet sound: %.1f dBFS < %.1f dBFS", level, VAD.min_level_db)
                    return False
                state.started_talking = True
                self.send_message_sync(create_message("log", "started_talking"))
                # 話し始めを検出した区切り + その前の区切り（言い始め）を残す
                state.stream = np.concatenate(list(state.window)[-(VAD.preroll_chunks + 1) :])
            return False

        state.stream = np.concatenate((state.stream, audio))
        state.silence_s = state.silence_s + len(audio) / sampling_rate if speech_s < VAD.silence_speech_s else 0.0
        return state.silence_s >= VAD.end_silence_s or len(state.stream) / sampling_rate >= VAD.max_utterance_s
