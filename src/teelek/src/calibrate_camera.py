import cv2
import numpy as np
import yaml
import time

# ================= CONFIG =================
chessboard_size = (8, 5)      # จำนวน inner corners (ช่องลบ 1)
square_size = 0.03            # ขนาดช่องจริง (เมตร)
required_frames = 75          # จำนวนเฟรมที่ต้องการเก็บ

# ================= PREPARE OBJECT POINTS =================
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0],
                       0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

# ================= CAMERA =================
W = 640
H = 480

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Camera resolution:",
      cap.get(cv2.CAP_PROP_FRAME_WIDTH),
      cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Camera FPS:",
      cap.get(cv2.CAP_PROP_FPS))

# ================= SUBPIX CRITERIA =================
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

last_capture_time = 0
capture_delay = 0.7

print("\n=== AUTO CALIBRATION START ===")
print("ขยับกระดานหลายมุม: เอียง, ใกล้, ไกล, ซ้าย, ขวา")
print("ระบบจะเก็บภาพอัตโนมัติ")
print("กด q เพื่อออก\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_cb, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret_cb:
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret_cb)

        current_time = time.time()

        if current_time - last_capture_time > capture_delay:
            objpoints.append(objp)
            imgpoints.append(corners2)
            last_capture_time = current_time
            print(f"Captured {len(objpoints)}/{required_frames}")

    cv2.putText(frame, f"Captured: {len(objpoints)}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Auto Calibration", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if len(objpoints) >= required_frames:
        print("\nEnough frames captured. Calibrating...")
        break

cap.release()
cv2.destroyAllWindows()

# ================= CALIBRATION =================
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None,
    flags=cv2.CALIB_RATIONAL_MODEL
)

# ================= REPROJECTION ERROR =================
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )
    error = cv2.norm(imgpoints[i],
                     imgpoints2,
                     cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)

# ================= RESULT =================
print("\n=== CALIBRATION RESULT ===")
print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)
print("\nMean Reprojection Error:", mean_error)

if mean_error < 0.2:
    print("Calibration Quality: EXCELLENT")
elif mean_error < 0.4:
    print("Calibration Quality: GOOD")
else:
    print("Calibration Quality: NEED IMPROVEMENT")

# ================= SAVE YAML =================
data = {
    "camera_matrix": camera_matrix.tolist(),
    "dist_coeff": dist_coeffs.tolist()
}

with open("camera_calibration.yaml", "w") as f:
    yaml.dump(data, f)

print("\nSaved to camera_calibration.yaml")