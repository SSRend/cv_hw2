import cv2
import numpy as np

# 파일 목록 (입력 → 출력)
file_pairs = [
    ("image_statue.png", "output_statue.png"),
    ("image_wave.png", "output_wave.png")
]

for input_path, output_path in file_pairs:
    # 이미지 불러오기
    img = cv2.imread(input_path)

    # Grayscale 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Median Blur 적용 (노이즈 제거)
    gray = cv2.medianBlur(gray, 5)

    # Adaptive Threshold를 이용한 외곽선 검출
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

    # **외곽선 두껍게 만들기 (Dilation)**
    kernel = np.ones((3,3), np.uint8)  
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 양방향 필터 적용 (뭉개지는 효과 방지)
    color = cv2.bilateralFilter(img, 9, 100, 100)

    # 샤프닝 필터 적용 (선명한 효과)
    sharpen_kernel = np.array([[-1, -1, -1], 
                               [-1,  9, -1], 
                               [-1, -1, -1]])
    color = cv2.filter2D(color, -1, sharpen_kernel)

    # 색상 이미지와 외곽선 결합
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # 결과 저장
    cv2.imwrite(output_path, cartoon)
    print(f"Saved: {output_path}")

# 모든 작업 완료 후 종료
print("✅ 모든 이미지 처리가 완료되었습니다!")
