data <- read.csv(file="alex_05lr.csv", header=TRUE, sep=",")

plot(data$Step, data$Value,type="l",xlab="Steps",
     ylab="Accuracy",main="Custom AlexNet 0.05LR Evaluation Accuracy",cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
