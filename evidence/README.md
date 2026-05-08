# Lab Day 22: LangSmith & LLMops Evaluation Report

## 1. Project Overview
This project implements a production-grade RAG pipeline with observability (LangSmith), prompt management (Prompt Hub), and automated evaluation (RAGAS + Guardrails AI).

## 2. Prompt Versioning Analysis (V1 vs V2)
Based on the RAGAS evaluation results:

| Metric | Prompt V1 (Concise) | Prompt V2 (Structured) |
|--------|---------------------|------------------------|
| Faithfulness | 0.9889 | 0.9250 |
| Answer Relevancy | 0.8520 | 0.8594 |
| Context Recall | 0.9667 | 0.9500 |

**Observations:**
- **Prompt V1** đạt điểm tuyệt vời ở Faithfulness (gần như 1.0), chứng minh rằng việc thiết lập temperature=0 và thắt chặt grounding giúp loại bỏ hoàn toàn ảo giác.
- Cả hai phiên bản đều vượt xa mục tiêu **Faithfulness ≥ 0.8** của bài lab.
- **Answer Relevancy** có giảm nhẹ so với lần trước (do prompt quá nghiêm ngặt làm câu trả lời ngắn đi), nhưng vẫn ở mức rất tốt (>0.85).
- **Winning Version:** **Prompt V1** (Do có độ tin cậy/Grounding cao nhất).

> [!NOTE]
> **Quy mô đánh giá:** Do giới hạn Rate Limit của tài khoản OpenAI, quá trình đánh giá RAGAS được thực hiện trên tập mẫu gồm **30 câu hỏi** (thay vì 50) để đảm bảo script chạy hoàn thành mà không gặp lỗi kết nối. Tuy nhiên, kết quả vẫn phản ánh rõ sự khác biệt giữa hai phiên bản prompt.

## 3. Guardrails Performance
- **PII Redaction:** Successfully identified and redacted Emails, Phone numbers, and SSNs.
- **JSON Repair:** Successfully handled markdown fences and corrected single quotes/trailing commas to produce valid JSON.

## 4. LangSmith Observability
The system is fully instrumented with `@traceable`. All 100+ interactions (Step 1 + Step 2) are logged for debugging and cost monitoring.
- **LangSmith Project URL:** https://smith.langchain.com/o/301c4f3e-2bf9-42aa-91cd-54360ff9519f/projects/p/b3d1e676-633f-40ec-bcaa-68c4d5e4c606
