from fastai.vision.all import *
from pathlib import Path

def main():
    # 1. Point at your data
    path = Path('data')
    dls  = ImageDataLoaders.from_folder(
        path,
        train='train',
        valid='valid',
        item_tfms=Resize(224),
        batch_tfms=aug_transforms()
    )

    # 2. Create the learner
    learn = vision_learner(dls, resnet34, metrics=accuracy)

    # 3. Fine-tune (reproduces your previous training)
    learn.fine_tune(5, base_lr=1.32e-02)

    # 4. Export to disk
    learn.export('bird_classifier.pkl')
    print("Exported bird_classifier.pkl")

if __name__ == '__main__':
    main()

