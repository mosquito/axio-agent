# axio-transport-google

Google GenAI (Gemini) transport for Axio with image and video support.

## API reference

Request/response types are generated from the Vertex AI discovery documents:

- **v1:** https://aiplatform.googleapis.com/$discovery/rest?version=v1
- **v1beta1:** https://aiplatform.googleapis.com/$discovery/rest?version=v1beta1

To regenerate TypedDict definitions after an API update:

```bash
python scripts/generate_types.py [--version v1beta1]
```

This produces `src/axio_transport_google/_generated_types.py`, which the transport
uses for type annotations. Conformance tests in `tests/test_generated_types.py`
validate that payload builders match the discovery schema.
