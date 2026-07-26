"""
reuse_rag — Module M3 (biến thể): API-grounded Reuse RAG
========================================================
Bài toán: cho một project Python cho sẵn, khi prompt yêu cầu làm một việc mà
project ĐÃ CÓ API để làm, model có TÁI SỬ DỤNG đúng API đó thay vì viết lại không?

Thí nghiệm chính: giữ retrieval cố định (oracle — nạp thẳng API đích), CHỈ thay đổi
"input representation" (độ giàu của context) qua 4 mức L1..L4, đo reuse-rate.
→ Đường cong "context giàu bao nhiêu thì reuse tăng bấy nhiêu".

Project corpus: boltons (thư viện utility Python thật trên GitHub, pip-installable).
"""
