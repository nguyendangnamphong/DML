# Double Machine Learning                                                          
**1) Kiểm tra**                              
**a) Ni là biến điều trị**                                                           
**Thông tin**                                                                                   
+ Biến nhiễu (confounders): 29 biến (Tất cả các cột số trừ Ni (biến điều trị) và Electrical_Conductivity_IACS (biến kết quả)).                                  
+ Số lượng mẫu: 1690 mẫu.                                                                                              
+ Biến điều trị Ni: 59 (distinct values); 3.55 (mean); 2.32 (độ lệch chuẩn); khoảng từ 0 đến 8.                                              
+ Biến kết quả Electrical_Conductivity_IACS: 85 (distinct values); 39.18 (mean); 15.08 (độ lệch chuẩn); khoảng từ 4 đến 92.
                                                                                
                                 
**Kết quả**                                                  
<img width="1613" height="830" alt="Ni" src="https://github.com/user-attachments/assets/1a1e0458-9f30-4950-a14b-c342d1344fec" />
+ Hiệu ứng nhân quả trung bình(ATE): 14,41% ( Ni tăng 1% thì độ dẫn điện tăng 14.41%; điều này mâu thuẫn với thực tế)
+ Khoảng tin cậy rộng: từ - 496.13 đến 1311.92 ( độ chắc chắn không cao và không ổn định)
+ Thông báo: thư viện econml có cảnh báo "Co-variance matrix is underdetermined. Inference will be invalid!", do đó có thể kết luận suy luận này không đáng tin cậy.                                          
                                                     
**Vấn đề có khả năng liên quan**                                   
+ Ma trận hiệp phương sai không đủ điều kiện: có 29 biến nhiễu mà chỉ có 1690 mẫu.
+ Outlier: Có ngoại lại ở biến kết quả, chi tiết xem ở biểu đồ tại [Link github của Data](https://github.com/nguyendangnamphong/data).
+ Mô hình chưa đủ khả năng.
                                                                                    
 **b) Placebo Test**                                 
 **Thông tin**                                  
 + Placebo Treatment: Solid_Solution_Temp_K (nhiệt độ dung dịch rắn)
 + Số lượng mẫu: 1690 mẫu.
 + Solid_Solution_Temp_K: 73 (distinct values); 6.32 (mean); 18.81 (độ lệch chuẩn); khoảng từ 0 đến 408.
 + Electrical_Conductivity_IACS: 85 (distinct values); 39.18 (mean); 15.08 (độ lệch chuẩn); khoảng từ 4 đến 92.
                                                                                   
**Kết quả**                                                                          
<img width="1603" height="822" alt="Capture" src="https://github.com/user-attachments/assets/93ac33d1-cb19-46d4-a0a1-b80c6d626ed0" />                                                                              
+ ATE: 0.5960692243818116 (thời gian lão hóa tăng thêm 1 giờ thì độ dẫn điện tăng 0.59 %IACS) (Khá gần 0 nên thể hiện đúng việc nó không có tác động đáng kể)               
+ Khoảng tin cậy: 95% (giao động từ -1.289 đến 3.433)            (Bao gồm cả 0, thể hiện hiệu ứng nhân quả không có khác biệt đáng kể so về mặt thống kê)                                                                              
                                                
**Vấn đề**                                                                      
Có cảnh báo về ma trận hiệp phương sai:"Co-variance matrix is underdetermined. Inference will be invalid!".                                        

**c) Cải thiện**
+ Giảm biễn nhiễu
+ Xóa bỏ Outlier
+ Tùy chỉnh mô hình: Random Forest (Giảm max_depth hoặc tăng min_samples_split), DML (Thử CausalForestDML). 
