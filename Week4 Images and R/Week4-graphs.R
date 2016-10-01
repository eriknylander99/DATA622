library(scatterplot3d)

cols <- c('x', 'y', 'ev', 'pv')
MSE <- read.csv('C:/Users/Erik/Dropbox/DATA622/MSE_data.csv', header = FALSE,
                col.names = cols)

MSE['error'] <- MSE['ev'] - MSE['pv']

scatterplot3d(MSE$x, MSE$y, MSE$ev, highlight.3d=TRUE, col.axis="blue",
              col.grid="lightblue", main="Function Value", pch=19)

scatterplot3d(MSE$x, MSE$y, MSE$pv, highlight.3d=TRUE, col.axis="blue",
              col.grid="lightblue", main="Function Approximation", pch=19)

scatterplot3d(MSE$x, MSE$y, MSE$error, highlight.3d=TRUE, col.axis="blue",
              col.grid="lightblue", main="Error(Expected - Predicted)", pch=19)

MAV <- read.csv('C:/Users/Erik/Dropbox/DATA622/MAV_data.csv', header = FALSE,
                col.names = cols)

MAV['error'] <- MAV['ev'] - MAV['pv']

scatterplot3d(MAV$x, MAV$y, MAV$ev, highlight.3d=TRUE, col.axis="blue",
              col.grid="lightblue", main="Function Approximation", pch=19)

scatterplot3d(MAV$x, MAV$y, MAV$pv, highlight.3d=TRUE, col.axis="blue",
              col.grid="lightblue", main="Function Approximation", pch=19)

scatterplot3d(MAV$x, MAV$y, MAV$error, highlight.3d=TRUE, col.axis="blue",
              col.grid="lightblue", main="Error(Expected - Predicted)", pch=19)

