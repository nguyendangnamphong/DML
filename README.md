# Double Machine Learning                  
**1) Sử dụng data là filtered_data**                                            
                                         
**a) Ni là biến điều trị**                                                           
**Thông tin**                                                                                   
+ Biến nhiễu (confounders): 28 biến( Tất cả chất trừ Ni(biến điều trị) và Electrical_Conductivity_IACS(biến kết quả) ).                                  
+ Số lượng mẫu: 1690 mẫu.                                                                                              
+ Biến điều trị Ni: 59 (distinct values); 3.55 (mean); 2.32 (độ lệch chuẩn); khoảng từ 0 đến 8.                                              
+ Biến kết quả Electrical_Conductivity_IACS: 85 (distinct values); 39.18 (mean); 15.08 (độ lệch chuẩn); khoảng từ 4 đến 92.
                                                                                
                                 
**Kết quả**                                                  
<img width="1613" height="830" alt="Ni" src="https://github.com/user-attachments/assets/1a1e0458-9f30-4950-a14b-c342d1344fec" />
+ Hiệu ứng nhân quả trung bình(ATE): 14,41% ( Ni tăng 1% thì độ dẫn điện tăng 14.41%; điều này không đúng với thực tế, vì theo [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1003632621655292) thì việc tăng Ni sẽ làm tăng kết tủa và giảm dẫn điện)
+ Khoảng tin cậy rộng: từ - 496.13 đến 1311.92 ( độ chắc chắn không cao và không ổn định)                                                       
+ Cảnh báo: "Co-variance matrix is underdetermined. Inference will be invalid!".                                          
                                                     
                                                                                    
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
+ Cảnh báo:"Co-variance matrix is underdetermined. Inference will be invalid!".                                                                                    
                                                                            
**c)Vấn đề**                                   
+ Ma trận hiệp phương sai không đủ điều kiện: có 29 biến (28 biến nhiễu và 1 biến điều trị) mà chỉ có 1690 mẫu.
+ Outlier: Trong Electrical_Conductivity_IACS, có 104 cái là outlier.
                                                        
                                     
**d) Ý tưởng cải thiện**
+ Giảm dữ liệu: Giảm từ 29 chất xuống còn 8 chất (Cu, Ni, Si, Al, Mg, Ti, B, Zr) (Nguyên nhân chọn 8 chất trên, em đã giải thích trong phần "III) Phân tích dữ liệu mới" của [Báo cáo dữ liệu](https://docs.google.com/document/d/10M1T0ri0ObnZa7OiIkymtEvvfM4knWlq1v-2LZeKW6w/edit?tab=t.0#heading=h.dvri5th2zbbo) ) 
+ Xóa bỏ Outlier: xóa đi 104 trong 1690 mẫu
+ Tùy chỉnh mô hình: Random Forest (Giảm max_depth hoặc tăng min_samples_split), DML (Thử CausalForestDML).
                            
**2) Sử dụng data là final_data**                                                 
                                           
 **a) Khác biệt với data cũ**                                                    
 + Rút gọn: giảm từ 29 chất xuống còn 6 chất(Cu, Al, Cr, Mg, Ni, Si)
 + Tiêu chí chọn: bỏ tất cả các chất có số lượng giá trị 0 lớn hơn 1500
 + Chi tiết số lượng giá trị 0 tại mỗi cột xem tại: vấn đề có nhiều giá trị 0 ở README.md của [Data](https://github.com/nguyendangnamphong/data).
                                                           
**b) Kết quả mô hình mới**                                                                     
+ Không còn cảnh báo "Co-variance matrix is underdetermined. Inference will be invalid!" nữa.
+ Hiệu ứng nhân quả trung bình (ATE): 30.43633356736651 (tức tăng 1% Ni thì độ dẫn điện tăng 30.436%; điều này không đúng với thực tế, vì theo [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1003632621655292) thì việc tăng Ni sẽ làm tăng kết tủa và giảm dẫn điện) 
+ Khoảng tin cậy: giao động từ -18 đến 149                                                                                                                             
<img width="1000" height="600" alt="causal_effect_plot" src="https://github.com/user-attachments/assets/5c236b93-e7e6-4370-ae18-d77b2b76cb14" />
                                                                                          
**c) Placebo Test**                                                   
+ Không còn cảnh báo "Co-variance matrix is underdetermined. Inference will be invalid!" nữa.
+ Hiệu ứng nhân quả trung bình (ATE) là: 0.0004971034671182595 (khá gần 0)
+ Khoảng tin cậy: giao động từ -0.0002 đến 0.0053 (bao gồm cả 0)                                                                                                                          
<img width="1000" height="600" alt="placebo_effect_plot" src="https://github.com/user-attachments/assets/434b80ee-72be-481c-95a8-60836f9bd50b" />


