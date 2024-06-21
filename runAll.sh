# First generate the gender_agreement dataset
python3 data_creation/gender_agreement.py

# Fine-tune models on the gender_agreement dataset
python3 fine_tuning/train.py

# We use the following library to calculate context mixing scores
cd cm_calculation/
git clone https://github.com/hmohebbi/context_mixing_toolkit.git -q
cd ..

# Save the cue words context mixing scores
python3 cm_calculation/main.py