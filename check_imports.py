print("Importing numpy...")
import numpy
print(f"Numpy version: {numpy.__version__}")

try:
    print("Importing cv2...")
    import cv2
    print("cv2 success")
except Exception as e:
    print(f"cv2 failed: {e}")

try:
    print("Importing sklearn...")
    from sklearn.covariance import EmpiricalCovariance
    print("sklearn success")
except Exception as e:
    print(f"sklearn failed: {e}")

try:
    print("Importing pandas...")
    import pandas
    print("pandas success")
except Exception as e:
    print(f"pandas failed: {e}")
