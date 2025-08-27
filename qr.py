import cv2

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

qr_detector = cv2.QRCodeDetector()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # # 그레이스케일 변환
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray', gray)
    #
    # # 에지 검출
    # edges = cv2.Canny(frame, 100, 200)
    # cv2.imshow('Edge', edges)
    #
    # # 카툰 효과
    # color = cv2.bilateralFilter(frame, 9, 75, 75)
    # edge_cartoon = cv2.Canny(color, 100, 200)
    # edge_inv = cv2.bitwise_not(edge_cartoon)
    # edge_inv = cv2.cvtColor(edge_inv, cv2.COLOR_GRAY2BGR)
    # cartoon = cv2.bitwise_and(color, edge_inv)
    # cv2.imshow('Cartoon', cartoon)
    #
    # # 원본 영상
    # cv2.imshow('view', frame)

    # QR 코드 검출 및 디코딩
    data, bbox, _ = qr_detector.detectAndDecode(frame)
    if bbox is not None:
        bbox = bbox.astype(int)
        n = len(bbox[0])
        for i in range(n):
            pt1 = tuple(bbox[0][i])
            pt2 = tuple(bbox[0][(i+1) % n])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    if data:
        cv2.putText(frame, data, (bbox[0][0][0], bbox[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('QR Reader', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()