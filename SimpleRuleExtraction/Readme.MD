This is a simplier version of rule extrarction where we focus on 5 parts:
1. Equipemnt
Type: Dumbbell, Barbell, Ball... 
2. Body
Position: Upright, horizontal/laying down, Sitted
3. Legs
Positon: Up, Down, Left, Right
Movement: Yes/No
4. Arms
Position: Up Down Left Right
Movement: Yes/No

Couple things: 
1. Movement is determined based on the change of angles over a persiod of frames. No movement should be reported if the angle changes is less than a theshold for that persiod

2. Position of legs and arms should be determined by comparing against the body gravity line

Modules:
1. Yolov7 for object detection
2. RuleExtraction Module that has different sections:
    Mediapipe pose estimation
    Butterworth denoising filter
    dense optical flow (?)

Goal:
Provide accurate rule extraction so that we can do clustering based on the rules extracted, the clusters shall then each train a smaller model for it.