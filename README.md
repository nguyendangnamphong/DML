# Double Machine Learning                                                          
**1) Nhân quả của Ni với độ dẫn điện**                              
**a) Nguyên bản**                                                           
**Thông tin**                                                                                   
+ Biến nhiễu (confounders): 29 biến (Tất cả các cột số trừ Ni (biến điều trị) và Electrical_Conductivity_IACS (biến kết quả)).                                  
+ Số lượng mẫu: 1690 mẫu.                                                                                              
+ Biến điều trị Ni: 59 (Số giá trị duy nhất); 3.55 (mean); 2.32 (độ lệch chuẩn); khoảng từ 0 đến 8.                                              
+ Biến kết quả Electrical_Conductivity_IACS: 39.18 (mean); 15.08 (độ lệch chuẩn); khoảng từ 4 đến 92.

                                 
**Kết quả**                                                  
<img width="1613" height="830" alt="Ni" src="https://github.com/user-attachments/assets/1a1e0458-9f30-4950-a14b-c342d1344fec" />
+ Hiệu ứng nhân quả trung bình: 14,41% ( Ni tăng 1% thì độ dẫn điện tăng 14.41%; điều này mâu thuẫn với thực tế)
+ Khoảng tin cậy rộng: từ - 496.13 đến 1311.92 ( độ chắc chắn không cao và không ổn định)
+ Thông báo: thư viện econml có cảnh báo "Co-variance matrix is underdetermined. Inference will be invalid!", do đó có thể kết luận suy luận này không đáng tin cậy.                                          
                                                     
**Vấn đề có khả năng liên quan**                                   
+ Ma trận hiệp phương sai không đủ điều kiện: có 29 biến nhiễu mà chỉ có 1690 mẫu.
+ Outlier: Có ngoại lại ở biến kết quả, chi tiết xem ở biểu đồ tại [Link github của Data](https://github.com/nguyendangnamphong/data).
+ Mô hình chưa đủ khả năng.
                                                            
**b) Cải thiện**
+ Giảm biễn nhiễu
+ Xóa bỏ Outlier
+ Tùy chỉnh mô hình: Random Forest (Giảm max_depth hoặc tăng min_samples_split), DML (Thử CausalForestDML). 
