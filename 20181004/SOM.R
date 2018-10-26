library(kohonen)
# import data, choose file interactively
data <- read.csv(file.choose(), header = T)

str(data)

X <- scale(data[, -1])

summary(X)

# unsupervised SOM
set.seed(224)

g <- somgrid(xdim = 10, ydim = 10)

map <- som(X, grid=g, alpha = c(0.05, 0.01), radius = 1)

plot(map, type = 'changes')

plot(map)

plot(map, type = 'mapping')

plot(map, type = 'count')

# supervised SOM
set.seed(222)
ind <- sample(2, nrow(X), replace = T, prob = c(0.7, 0.3))
train <- data[ind == 1,]
test <- data[ind == 2,]

trainX <- scale(train[, -1])
testX <- scale(test[, -1], 
               center = attr(trainX, "scaled:center"), 
               scale = attr(trainX, "scaled:scale"))
trainY <- factor(train[, 1])
Y <- factor(test[, 1])
test[,1] <- 0
testXY <- list(independent = testX, dependent = test[,1])

# classification and prediction model

set.seed(223)
map1 <- xyf(trainX, 
            classvec2classmat(factor(trainY)),
            grid = somgrid(10, 10, "hexagonal"),
            rlen = 200)

plot(map1, type = 'changes')

plot(map1)

pred <- predict(map1, newdata = testXY)

table(Predicted = pred$predictions[[2]], Actual = Y)
