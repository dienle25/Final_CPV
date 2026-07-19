# Scope and pipeline

## Included

- Bundled YOLOv8s three-class checkpoint.
- Local MP4 and webcam input.
- ByteTrack rider IDs.
- One-to-one rider/head association.
- Temporal majority voting and event confirmation.
- Full-frame evidence, SQLite and CSV.
- Streamlit interface.
- Optional experimental PaddleOCR and SMTP.

## Excluded

- RTSP/network stream handling.
- React, Django, Redis and cloud deployment.
- Autonomous legal enforcement.
- Claims of validated plate recognition.

## End-to-end flow

```text
MP4/webcam
-> YOLOv8s [helmet, no_helmet, rider]
-> ByteTrack rider ID
-> geometric one-to-one head association
-> temporal majority label
-> repeated stable no_helmet status
-> evidence frame + SQLite + CSV
-> annotated video / Streamlit
```

The OCR branch is optional because the bundled checkpoint has no license-plate
class. When enabled, it runs once per confirmed event on a heuristic crop and
may return `UNREAD`.
