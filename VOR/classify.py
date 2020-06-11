

import cv2
import numpy as np
import os
import Train
import ImageExtract


def classifier_classify(image):
    model_path = os.path.join("data/model/", "classifier.model")
    model = Train.ElementClassifier(model_path=model_path, load=True)
    labels = []
    bboxes = ImageExtract.extract_elements(image)
    for bbox in bboxes:
        x, y, w, h = bbox
        roi = image[y - 20:y + h + 20, x - 20:x + w + 20]

        if np.size(roi) == 0:
            continue

        pred = model.predict(roi, preprocess=True)
        indx = np.argmax(pred, axis=1)[0]
        score = pred[0][indx]
        cv2.rectangle(image, (x-20, y-20), (x + w+20, y + h+20), (0, 191, 255), thickness=2)
        text = str(indx) + ":" + model.id2label[indx] + ":" + str(score)
        #cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
        #cv2.imshow("Input", image)
        #print(indx, model.id2label[indx], score)
        labels.append(model.id2label[indx])
    
    return labels    
    cv2.waitKey(0)
    
