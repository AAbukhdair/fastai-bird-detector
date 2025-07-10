import cv2
from fastai.vision.all import load_learner, PILImage

THRESHOLD = 0.5
learn = load_learner('data/bird_classifier.pkl')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = PILImage.create(frame)
    pred, idx, probs = learn.predict(img)

    # only overlay if above threshold
    if probs[idx] > THRESHOLD:
        label = f'{pred} ({probs[idx]*100:.1f}%)'
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result _inside_ the while
    cv2.imshow('Bird Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# once we break out of the loop, clean up
cap.release()
cv2.destroyAllWindows()
