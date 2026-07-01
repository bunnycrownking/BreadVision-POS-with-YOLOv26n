import cv2
import numpy as np
import os

# 1. พิกัดจริงบนโต๊ะ (Real-world Ground Centimeters)
# กำหนดให้จุดศูนย์กลางคือ (0,0) และลำดับต้องตรงกับตอนคลิก source points
dst_pts = np.array([
    [0, 0],       # จุดศูนย์กลาง (cm, cm)
    [0, 18.5],    # จุดตัดบน (0, r)
    [0, -18.5],   # จุดตัดล่าง (0, -r)
    [18.5, 0],    # จุดตัดขวา (r, 0)
    [-18.5, 0]    # จุดตัดซ้าย (-r, 0)
], dtype=np.float32)

# รายชื่อไฟล์และกล้องที่ต้องประมวลผล
image_infos = [
    {'path': './check/Top.jpg', 'name': 'Top Camera'},
    {'path': './check/Side1.jpg', 'name': 'Side1 Camera'},
    {'path': './check/Side2.jpg', 'name': 'Side2 Camera'}
]

# ตัวแปรเก็บพิกัด
all_src_points = {}
current_points = []
img_display = None

def click_event(event, x, y, flags, param):
    global current_points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < 5:
            current_points.append([x, y])
            print(f"คลิกจุดที่ {len(current_points)}: พิกัด ({x}, {y})")
            # วาดจุดสีแดง
            cv2.circle(img_display, (x, y), 2, (0, 0, 255), -1)
            cv2.imshow(param, img_display)

print("=== โปรแกรมคำนวณ Homography Matrix สำหรับ 3 กล้อง ===")
print("กรุณาคลิกเลือกจุดอ้างอิงให้ครบ 5 จุดในแต่ละภาพตามลำดับที่วางแผนไว้")
print("หากต้องการยกเลิกกลางคัน ให้กดปุ่ม 'q'\n")

# ขั้นตอนการคลิกจุดบนภาพทั้ง 3 ภาพ
for info in image_infos:
    img_path = info['path']
    cam_name = info['name']
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ข้อผิดพลาด] ไม่พบไฟล์ภาพ: {img_path}")
        continue
    
    img_display = img.copy()
    current_points = []
    
    window_name = f"Image: {cam_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) # ย่อขนาดหน้าต่างให้เล็กลงเวลาแสดงผล (ไม่กระทบพิกัดจริงของภาพ)
    cv2.imshow(window_name, img_display)
    cv2.setMouseCallback(window_name, click_event, param=window_name)
    
    print(f"-> เปิดภาพ {cam_name} ({img_path}) รอรับการคลิก 5 จุด...")
    
    # วนลูปรอจนกว่าจะคลิกครบ 5 จุด หรือกด q เพื่อออก
    while True:
        key = cv2.waitKey(10) & 0xFF
        if len(current_points) == 5:
            print(f"เก็บจุดของ {cam_name} ครบ 5 จุดแล้ว")
            cv2.waitKey(500) # รอสักพักให้เห็นจุดที่ 5 ถูกวาดลงไป
            break
        elif key == ord('q'):
            print("ยกเลิกการทำงาน")
            cv2.destroyAllWindows()
            exit(0)
            
    cv2.destroyWindow(window_name)
    
    # บันทึกภาพที่มีการวาดจุดแล้ว (ไม่ทับไฟล์เดิม)
    base, ext = os.path.splitext(img_path)
    save_path = f"{base}_marked{ext}"
    cv2.imwrite(save_path, img_display)
    print(f"บันทึกรูปภาพที่มาร์คจุดแล้วเป็น: {save_path}")
    
    # แปลงให้เป็น Numpy Array สำหรับการคำนวณ Homography
    all_src_points[cam_name] = np.array(current_points, dtype=np.float32)

cv2.destroyAllWindows()

print("\n=========================================")
print("          ผลลัพธ์ Homography Matrix")
print("=========================================")

# คำนวณ Homography และแสดงผลสรุปแยกกัน
for cam_name, src_pts in all_src_points.items():
    if len(src_pts) == 5:
        # คำนวณ Homography Matrix
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        
        print(f"\n--- {cam_name} ---")
        print("พิกัด Source Points (บนภาพพิกเซล):")
        for i, pt in enumerate(src_pts):
            print(f"  จุดที่ {i+1}: ({pt[0]}, {pt[1]})")
            
        if H is not None:
            print("Homography Matrix:")
            # จัดรูปแบบให้อยู่ในรูป array ของ numpy และมีเครื่องหมาย , คั่น
            formatted_H = np.array2string(H, separator=', ')
            print(f"np.array({formatted_H}, dtype=np.float32)")
        else:
            print("ไม่สามารถคำนวณ Homography Matrix ได้")