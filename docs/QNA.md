# Defense Q&A

1. **Why YOLOv8s?**  
   The supplied trained checkpoint is YOLOv8s with 11.14M parameters. It offers
   a practical accuracy/speed balance. The report names the real architecture
   instead of relabeling it as YOLOv8n.

2. **Why track riders instead of no-helmet boxes?**  
   Rider boxes are larger and more stable. Small head boxes are more likely to
   disappear during blur or occlusion.

3. **How is a head assigned to a rider?**  
   The head center must lie in the upper rider region. Candidate pairs are
   scored geometrically, then greedily assigned one-to-one.

4. **Why temporal voting?**  
   It reduces frame-to-frame class flicker. A second multi-frame gate prevents a
   single stable-looking frame from creating an event.

5. **How are duplicates prevented?**  
   One ByteTrack rider ID can be committed only once while that track state is
   active.

6. **What happens when a helmet is detected after no-helmet votes?**  
   An uncommitted violation candidate is reset. Old votes also expire after a
   configurable timeout.

7. **Is mAP@50 = 0.93 the system accuracy?**  
   No. It is a YOLO bounding-box metric on an internal test split. Tracking,
   association and event precision require separate video-level evaluation.

8. **Why SQLite?**  
   It is local, transactional and requires no server, which fits a classroom
   MVP better than a web backend.

9. **Does OCR work?**  
   OCR is optional and experimental because the checkpoint has no plate class.
   `UNREAD` is an accepted fallback. A dedicated plate detector is future work.

10. **Can this issue legal penalties automatically?**  
    No. It is a decision-support prototype and all events require human review.
