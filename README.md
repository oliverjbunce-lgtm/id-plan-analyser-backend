---
title: ID Plan Analyser API
emoji: 🚪
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
---

# Independent Doors — Plan Analyser API

FastAPI backend for YOLOv8 door detection from building plans.

## Endpoints

- `GET /health` — liveness check
- `POST /upload` — upload PDF, returns page thumbnails + suggested floor plan page
- `POST /analyse` — upload PDF + page number, returns door detections + annotated image
