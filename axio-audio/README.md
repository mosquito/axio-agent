# axio-audio

Microphone capture and speaker playback for Axio realtime agents. Provides
`Microphone`, `Speaker`, and `DuplexAudio` on top of sounddevice and NumPy.
A separately installed realtime transport connects the agent to the model.

## Installation

```bash
pip install axio-audio
```

Audio capture and playback require working PortAudio input/output devices.
Install a realtime transport, such as `axio-transport-openai` or
`axio-transport-google`, to build a voice agent.

## Capture audio

```python
import asyncio

from axio_audio import Microphone


async def main() -> None:
    async with Microphone(sample_rate=24000, chunk_ms=50) as microphone:
        async for block in microphone:
            print(block.media_type, len(block.data))
            break


asyncio.run(main())
```

This example opens the microphone. Chunks are `axio.AudioBlock` objects
containing PCM16 mono audio. Send them to an open `RealtimeAgent` using
`await agent.send(block)`.

## Play audio

Use `Speaker` as an async context manager and feed raw PCM16 bytes received
from the model:

```python
from axio.events import AudioOutputDelta
from axio_audio import Speaker


async def play(agent) -> None:
    async with Speaker(sample_rate=24000) as speaker:
        async for event in agent.events():
            if isinstance(event, AudioOutputDelta):
                await speaker.feed(event.data)
```

Match capture and playback sample rates to the transport's input and output
formats. `speaker.stop()` clears queued playback when the user interrupts.
Context managers close devices even when a task fails or is cancelled.

## Duplex audio

`DuplexAudio` opens one PortAudio stream for input and output, so both use the
same device clock. Use `mic_chunks()` for capture and `feed_speaker()` for
playback. Its defaults are 48 kHz and 20 ms chunks. A shared clock helps an
echo cancellation pipeline; it does not implement echo cancellation itself.

See the [realtime audio guide](https://github.com/mosquito/axio-agent/blob/main/docs/guides/realtime-audio.md)
for a complete voice agent and the
[API reference](https://github.com/mosquito/axio-agent/blob/main/docs/api/audio.md) for device
and buffering options.
