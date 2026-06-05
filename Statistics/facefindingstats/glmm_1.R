# Load the required package
library(lme4)

# Load the cleanly formatted CSV
data <- read.csv("all_info.csv")

# Explicitly convert categorical variables to factors
data$groupID <- as.factor(data$groupID)
data$groupSize <- as.factor(data$groupSize)
data$spatiality <- as.factor(data$spatiality)
data$environment <- as.factor(data$environment)
data$familiarity <- as.factor(data$familiarity)

# ---------------------------------------------------------
# MODEL 1: Zero-Face Attention (Entire Dataset)
# Tests the isolated effect of each condition on looking away
# ---------------------------------------------------------
model_zero <- glmer(
  cbind(zeroFaceFrames, analyzedFrames - zeroFaceFrames) ~ spatiality + environment + groupSize + familiarity + (1 | groupID), 
  family = binomial, 
  data = data
)

print("Summary for Zero-Face Attention:")
summary(model_zero)

# ---------------------------------------------------------
# MODEL 2: Single-Target Attention (Subset: Spatial 3 vs 4)
# Tests how adding a 4th person changes 1-on-1 focus
# ---------------------------------------------------------
# Isolate the data to only Spatial conditions with 3 or 4 people
data_spatial <- subset(data, spatiality == "Spatial" & groupSize %in% c("three", "four"))

model_single <- glmer(
  cbind(oneFaceFrames, analyzedFrames - oneFaceFrames) ~ groupSize + environment + familiarity + (1 | groupID), 
  family = binomial, 
  data = data_spatial
)

print("Summary for Single-Target Attention:")
summary(model_single)

# -----
# Model 3
# -----
# Isolate the data to Window conditions only
data_window <- subset(data, spatiality == "Window")

# Dynamically set the "maximum face frames" based on group size constraints in the Window condition
data_window$maxFaceFrames <- ifelse(
  data_window$groupSize == "two", data_window$oneFaceFrames,
  ifelse(data_window$groupSize == "three", data_window$twoFaceFrames,
         ifelse(data_window$groupSize == "four", data_window$threeFaceFrames, NA))
)

# Calculate disengaged frames (analyzed frames minus the max face frames)
data_window$disengagedFrames <- data_window$analyzedFrames - data_window$maxFaceFrames

# MODEL 3: Intentional Disengagement (Window Only)
# Tests the likelihood of looking at fewer than the maximum allowable faces
model_disengage <- glmer(
  cbind(disengagedFrames, maxFaceFrames) ~ environment + groupSize + familiarity + (1 | groupID), 
  family = binomial, 
  data = data_window
)

print("Summary for Intentional Disengagement in Window Conditions:")
summary(model_disengage)
