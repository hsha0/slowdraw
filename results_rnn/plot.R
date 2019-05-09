data <- read.csv(file="4cnn_2rnn_00005lr.csv", header=TRUE, sep=",")

plot(data$Step, data$Value,type="l",xlab="Steps",
     ylab="Accuracy",main="4CNN 2RNN 0.00005LR Evaluation Accuracy",cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)

data <- read.csv(file="3cnn_3rnn_0001lr.csv", header=TRUE, sep=",")

plot(data$Step, data$Value,type="l",xlab="Steps",
     ylab="Accuracy",main="3CNN 3RNN 0.0001LR Evaluation Accuracy",
     cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)