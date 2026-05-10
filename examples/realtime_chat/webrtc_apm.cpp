// Native Python binding to webrtc-audio-processing-1's AudioProcessing
// (the same APM/AEC3 stack that PipeWire's module-echo-cancel uses by
// default).  Built as a pybind11 extension so callers get a real
// Python class instead of an opaque ctypes void*.
//
// Build:
//   uv pip install pybind11           # build-time only
//   pkg-config webrtc-audio-processing-1 --cflags --libs
//   python -m pip install -e .        # via the local pyproject.toml
//
// Frame size is fixed at 10 ms by webrtc — caller must hand in
// ``sample_rate / 100`` samples per call.  See ``aec.py``'s
// ``WebRtcAECProcessor`` for a streaming wrapper that buffers
// arbitrary chunk sizes into 10 ms frames.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <modules/audio_processing/include/audio_processing.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using webrtc::AudioProcessing;
using webrtc::AudioProcessingBuilder;
using webrtc::StreamConfig;


class WebRtcAEC {
public:
    WebRtcAEC(int capture_sr,
              int render_sr,
              int capture_channels = 1,
              int render_channels = 1,
              bool noise_suppression = true,
              bool transient_suppression = true,
              bool high_pass_filter = true)
        : apm_(AudioProcessingBuilder().Create()),
          capture_sr_(capture_sr),
          render_sr_(render_sr),
          capture_channels_(capture_channels),
          render_channels_(render_channels) {
        if (!apm_) {
            throw std::runtime_error("AudioProcessingBuilder().Create() returned null");
        }

        AudioProcessing::Config config;
        config.echo_canceller.enabled = true;
        config.echo_canceller.mobile_mode = false;
        config.echo_canceller.enforce_high_pass_filtering = true;
        config.high_pass_filter.enabled = high_pass_filter;
        config.noise_suppression.enabled = noise_suppression;
        config.noise_suppression.level =
            AudioProcessing::Config::NoiseSuppression::kHigh;
        config.transient_suppression.enabled = transient_suppression;
        config.gain_controller1.enabled = false;
        config.gain_controller2.enabled = false;
        apm_->ApplyConfig(config);
    }

    ~WebRtcAEC() {
        if (apm_) apm_->Release();
    }

    WebRtcAEC(const WebRtcAEC&) = delete;
    WebRtcAEC& operator=(const WebRtcAEC&) = delete;

    // Process one 10 ms mic frame.  Accepts any object exposing the
    // Python buffer protocol (bytes, bytearray, memoryview, numpy
    // int16 array) and returns a fresh ``bytes`` of the same length
    // with AEC + NS + transient suppression applied.  We size by
    // ``total_bytes`` rather than ``info.size`` so raw ``bytes``
    // (itemsize=1) work the same as a numpy int16 array (itemsize=2).
    py::bytes process_stream(py::buffer mic) {
        py::buffer_info info = mic.request();
        const ssize_t total_bytes = info.itemsize * info.size;
        const int expected_bytes = capture_sr_ / 100 * capture_channels_ * 2;
        if (total_bytes != expected_bytes) {
            throw std::invalid_argument(
                "process_stream: expected " + std::to_string(expected_bytes) +
                " bytes (10ms int16 @ " + std::to_string(capture_sr_) + " Hz × " +
                std::to_string(capture_channels_) + " ch), got " +
                std::to_string(total_bytes));
        }
        const ssize_t n = total_bytes / 2;
        std::vector<int16_t> buf(static_cast<int16_t*>(info.ptr),
                                 static_cast<int16_t*>(info.ptr) + n);
        StreamConfig cfg(capture_sr_, capture_channels_);
        const int rc = apm_->ProcessStream(buf.data(), cfg, cfg, buf.data());
        if (rc != 0) {
            throw std::runtime_error("ProcessStream returned " + std::to_string(rc));
        }
        return py::bytes(reinterpret_cast<const char*>(buf.data()), total_bytes);
    }

    // Process one 10 ms reference frame (the audio that's about to
    // hit the speaker).  AEC3 uses this to model the echo path; no
    // useful return value, but we follow the same int16 buffer
    // convention for symmetry with ``process_stream``.
    void process_reverse_stream(py::buffer ref) {
        py::buffer_info info = ref.request();
        const ssize_t total_bytes = info.itemsize * info.size;
        const int expected_bytes = render_sr_ / 100 * render_channels_ * 2;
        if (total_bytes != expected_bytes) {
            throw std::invalid_argument(
                "process_reverse_stream: expected " + std::to_string(expected_bytes) +
                " bytes (10ms int16 @ " + std::to_string(render_sr_) + " Hz × " +
                std::to_string(render_channels_) + " ch), got " +
                std::to_string(total_bytes));
        }
        const ssize_t n = total_bytes / 2;
        std::vector<int16_t> buf(static_cast<int16_t*>(info.ptr),
                                 static_cast<int16_t*>(info.ptr) + n);
        StreamConfig cfg(render_sr_, render_channels_);
        apm_->ProcessReverseStream(buf.data(), cfg, cfg, buf.data());
    }

    // Hint AEC3 about the round-trip render → capture latency.
    // AEC3 has its own delay estimator so the value isn't critical,
    // but a roughly correct hint speeds up the initial convergence.
    void set_stream_delay_ms(int delay_ms) {
        apm_->set_stream_delay_ms(delay_ms);
    }

    int capture_frame_samples() const { return capture_sr_ / 100; }
    int render_frame_samples() const { return render_sr_ / 100; }

private:
    AudioProcessing* apm_;
    int capture_sr_;
    int render_sr_;
    int capture_channels_;
    int render_channels_;
};


PYBIND11_MODULE(webrtc_apm, m) {
    m.doc() = "Native binding to webrtc-audio-processing-1 (AEC3 + NS + HPF + TS).";

    py::class_<WebRtcAEC>(m, "WebRtcAEC")
        .def(py::init<int, int, int, int, bool, bool, bool>(),
             py::arg("capture_sr"),
             py::arg("render_sr"),
             py::arg("capture_channels") = 1,
             py::arg("render_channels") = 1,
             py::arg("noise_suppression") = true,
             py::arg("transient_suppression") = true,
             py::arg("high_pass_filter") = true,
             R"doc(Create an AEC3-based AudioProcessing instance.

Frame size is fixed at 10 ms by webrtc-audio-processing — every
``process_stream`` / ``process_reverse_stream`` call must pass exactly
``sample_rate / 100 * num_channels`` int16 samples.)doc")
        .def("process_stream", &WebRtcAEC::process_stream,
             py::arg("mic_pcm"),
             "Apply AEC3 + NS + HPF + TS to one 10 ms mic frame.")
        .def("process_reverse_stream", &WebRtcAEC::process_reverse_stream,
             py::arg("speaker_pcm"),
             "Feed one 10 ms speaker reference frame.")
        .def("set_stream_delay_ms", &WebRtcAEC::set_stream_delay_ms,
             py::arg("delay_ms"))
        .def_property_readonly("capture_frame_samples",
                               &WebRtcAEC::capture_frame_samples)
        .def_property_readonly("render_frame_samples",
                               &WebRtcAEC::render_frame_samples);
}
