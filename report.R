#Load Libraries
library("dplyr")
library("gtools")
library("gmodels")
library("ggplot2")
library("class")
library("tidyr")
library("lattice")
library("caret")
library("rmdformats")
library("knitr")
library("gridExtra")
library("corrplot")
library("data.table")
library("mlr3verse")
library("randomForest")

#Load dataset
dataset <- read.csv(file = "https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv", header = T, sep=",")
dim(dataset)
view(dataset)
str(dataset)
summary(dataset)
skim(dataset)
head(dataset,3)

DataExplorer::plot_bar(dataset, ncol = 3)
DataExplorer::plot_histogram(dataset, ncol = 3)
DataExplorer::plot_boxplot(dataset, by = "fatal_mi", ncol = 3)

corrplot(cor(dataset),method="color",addCoef.col="grey")

ggplot(dataset, mapping = aes(x=age,fill=fatal_mi,color=fatal_mi)) +
  geom_histogram(binwidth = 1,color="black") + 
  labs(x = "Age",y = "Frequency", title = "Heart Disease w.r.t. Age")

ggplot(dataset, aes(x=age, y=serum_creatinine)) + 
  geom_point(aes(col=fatal_mi))

dataset <- dataset %>%
  mutate(anaemia = factor(anaemia, levels = c(0,1), labels = c("False", "True")),
         diabetes = factor(diabetes, levels = c(0,1), labels = c("False", "True")),
         high_blood_pressure = factor(high_blood_pressure, levels = c(0,1), labels = c("False", "True")),
         sex = factor(sex, levels = c(0,1), labels = c("Female", "Male")),
         smoking = factor(smoking, levels = c(0,1), labels = c("False", "True")),
         fatal_mi = factor(fatal_mi, levels = c(0,1), labels = c("False", "True")))
str(dataset)
colSums(is.na(dataset))

prop.table(table(dataset$fatal_mi))
table(dataset$fatal_mi)


set.seed(212)
heart_task <- TaskClassif$new(id = "HeartFailure",
                               backend = dataset,
                               target = "fatal_mi",
                               positive = "True")
View(as.data.table(mlr_resamplings))
mlr_resamplings

cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(heart_task)

View(as.data.table(mlr_learners))
mlr_learners

lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")

lrn_baseline$param_set
lrn_cart$param_set

res <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_baseline,
                    lrn_cart),
  resampling = list(cv5)
), store_models = TRUE)
res$aggregate()

trees <- res$resample_result(2)
tree1 <- trees$learners[[1]]
tree1_rpart <- tree1$model
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)
plot(res$resample_result(2)$learners[[5]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[5]]$model, use.n = TRUE, cex = 0.8)

lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)
res_cart_cv <- resample(heart_task, lrn_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[5]]$model)

lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.016)

res <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

mlr_pipeops
lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
pl_xgb <- po("encode") %>>%
  po(lrn_xgboost)
res <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp,
                    pl_xgb),
  resampling = list(cv5)
), store_models = TRUE)
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

#Logistic Regression
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
pl_log_reg <- pl_missing %>>%
  po(lrn_log_reg)

res <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp,
                    pl_xgb,
                    pl_log_reg),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))


#Advanced: super learning
library("data.table")
library("mlr3verse")

set.seed(212) # set seed for reproducibility

heart_task <- TaskClassif$new(id = "HeartFailure",
                              backend = dataset,
                              target = "fatal_mi",
                              positive = "False")

cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(heart_task)

lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.016, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")

lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

pl_factor <- po("encode")

spr_lrn <- gunion(list(
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv", lrn_cart),
    po("learner_cv", lrn_cart_cp)
  )),
  pl_missing %>>%
    gunion(list(
      po("learner_cv", lrn_ranger),
      po("learner_cv", lrn_log_reg),
      po("nop")
    )),
  pl_factor %>>%
    po("learner_cv", lrn_xgboost)
)) %>>%
  po("featureunion") %>>%
  po(lrnsp_log_reg)
spr_lrn$plot()

res_spr <- resample(heart_task, spr_lrn, cv5, store_models = TRUE)
res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr")))

#Random Forest
set.seed(123)
train <- sample(nrow(dataset), nrow(dataset)*0.7)
heart_train <- dataset[train, ]
heart_test <- dataset[-train, ]

heart_train.forest <- randomForest::randomForest(fatal_mi~., data = heart_train, importance = TRUE)
heart_train.forest

heart_predict <- predict(heart_train.forest, heart_train)

plot(heart_train$fatal_mi, heart_predict, main = 'train', 
     xlab = 'fatal_mi', ylab = 'Predict')
abline(1, 1)

heart_predict <- predict(heart_train.forest, heart_test)

plot(heart_test$fatal_mi, heart_predict, main = 'test',
     xlab = 'fatal_mi', ylab = 'Predict')
abline(1, 1)

importance_heart <- data.frame(importance(heart_train.forest), check.names = FALSE)
head(importance_heart)

varImpPlot(heart_train.forest, n.var = min(5, nrow(heart_train.forest$importance)), 
           main = 'Top 5 - variable importance')

set.seed(123)
heart_train.cv <- replicate(5, rfcv(heart_train[-ncol(heart_train)], heart_train$fatal_mi, cv.fold = 5, step = 1.5), simplify = FALSE)
heart_train.cv

heart_train.cv <- data.frame(sapply(heart_train.cv, '[[', 'error.cv'))
heart_train.cv$heart <- rownames(heart_train.cv)
heart_train.cv <- reshape2::melt(heart_train.cv, id = 'heart')
heart_train.cv$heart <- as.numeric(as.character(heart_train.cv$heart))

heart_train.cv.mean <- aggregate(heart_train.cv$value, by = list(heart_train.cv$heart), FUN = mean)
head(heart_train.cv.mean, 5)

ggplot(heart_train.cv.mean, aes(Group.1, x)) +
  geom_line() +
  theme(panel.grid = element_blank(), panel.background = element_rect(color = 'black', fill = 'transparent')) +  
  labs(title = '',x = 'Number of fatal_mi', y = 'Cross-validation error')
