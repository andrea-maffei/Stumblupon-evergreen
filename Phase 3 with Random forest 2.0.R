#dev.off()
source("~/Desktop/Masters/Classes/Machine Learning/Data for Class/BabsonAnalytics.r")
set.seed(1234)
########################### Initial Random Forest to Test Effectiveness ################################
#Import
df_control<-read.csv("~/Desktop/Masters/Classes/Machine Learning/Projects/Evergreen - stumbleupon/Data/train.csv")

#Manage
df_control$alchemy_category<-as.factor(df_control$alchemy_category)
df_control$is_news<-as.factor(df_control$is_news)
df_control$lengthyLinkDomain<-as.factor(df_control$lengthyLinkDomain)
df_control$news_front_page<-as.factor(df_control$news_front_page)
df_control$hasDomainLink<-as.factor(df_control$hasDomainLink)
df_control$label<-as.factor(df_control$label)
df_control$framebased<-as.factor(df_control$framebased)

df_control$url<-NULL
df_control$urlid<-NULL
df_control$boilerplate<-NULL
df_control$alchemy_category_score<-NULL

#Partition
N_control<- nrow(df_control)
training_size_control<-round(N_control*0.6) 
training_cases_control<-sample(N_control, training_size_control) 
training_control<-df_control[training_cases_control, ]
test_control<-df_control[-training_cases_control, ]

#Build, Predict, Evaluate
library(randomForest)#Bagging
model_control<-randomForest(label ~., data=training_control, ntree=750)
pred_control<- predict(model_control, test_control)
error_control<-sum(pred_control != test_control$label)/nrow(test_control)

########################### Classification Tree ################################

library(rpart)
library(rpart.plot)

#LOAD
df_tree<-read.csv("~/Desktop/Masters/Classes/Machine Learning/Projects/Evergreen - stumbleupon/Data/train.csv")
str(df_tree)

#MANAGE
#df<-subset(df, alchemy_category_score !="?") I don't think to use this

df_tree$alchemy_category<-as.factor(df_tree$alchemy_category)
df_tree$is_news<-as.factor(df_tree$is_news)
df_tree$lengthyLinkDomain<-as.factor(df_tree$lengthyLinkDomain)
df_tree$news_front_page<-as.factor(df_tree$news_front_page)
df_tree$hasDomainLink<-as.factor(df_tree$hasDomainLink)
df_tree$label<-as.factor(df_tree$label)
df_tree$framebased<-as.factor(df_tree$framebased)

df_tree$url<-NULL
df_tree$urlid<-NULL
df_tree$boilerplate<-NULL
df_tree$alchemy_category_score<-NULL


#PARTITION
N_tree<-nrow(df_tree) 
training_size_tree<-round(N_tree*0.6) 
training_cases_tree<-sample(N_tree, training_size_tree)
training_tree<-df_tree[training_cases_tree, ]
test_tree<-df_tree[-training_cases_tree, ]

#BUILD
stopping_rules=rpart.control(minsplit = 50, minbucket = 10, cp=0.01) 
model_tree<-rpart(label ~., data<-training_tree, )#control = stopping_rules)

model_tree<-easyPrune(model_tree)
rpart.plot(model_tree)

#PREDICT
predictions_tree<-predict(model_tree, test_tree, type = "class")

#EVALUATE
observations<-test_tree$label
table(predictions_tree, observations)
error_bench_tree<-benchmarkErrorRate(training_tree$label, test_tree$label)
errorRate_tree<-sum(predictions_tree!=test_tree$label)/nrow(test_tree)

########################### Logistic Regression ###############################

library(ggplot2)
library(caret)

#LOAD
df_glm<-read.csv("~/Desktop/Masters/Classes/Machine Learning/Projects/Evergreen - stumbleupon/Data/train.csv")
str(df_glm)

df_glm$label= as.logical(df_glm$label)
df_glm$boilerplate=NULL
df_glm$url=NULL
df_glm$urlid=NULL
df_glm$alchemy_category=as.factor(df_glm$alchemy_category)
df_glm$framebased=NULL
df_glm$alchemy_category_score<-NULL
df_glm$hasDomainLink=as.factor(df_glm$hasDomainLink)
df_glm$is_news=as.factor(df_glm$is_news)
df_glm$lengthyLinkDomain=as.factor(df_glm$lengthyLinkDomain)
df_glm$news_front_page=as.factor(df_glm$news_front_page)

set.seed(1234)
N_glm= nrow(df_glm)
trainingSize_glm = round(N_glm*0.6)
trainingCases_glm= sample(N_glm, trainingSize_glm)
training_glm= df_glm[trainingCases_glm,]
test_glm = df_glm[-trainingCases_glm,]

model_glm = glm(label ~., data=training_glm, family=binomial)


model_glm = step(model_glm)
summary(model_glm)
pred = predict(model_glm, test_glm, type='response')
predTF = (pred > 0.6)
table(predTF, test_glm$label)
errorRate_glm = sum(predTF != test_glm$label)/nrow(test_glm)
errorBench_glm = benchmarkErrorRate(training_glm$label, test_glm$label)

ROCChart(test_glm$label, pred)
liftChart(test_glm$label,pred)

############ Text Processing and Bag of Words for Naïve Bayes ###################
library(readr)
library(tidytext)
library(tidyverse)

df_txt = read.csv('~/Desktop/Masters/Classes/Machine Learning/Projects/Evergreen - stumbleupon/Data/train.csv')
df_txt$boilerplate = as.character(df_txt$boilerplate)
df_txt = df_txt %>% unique()
df_txt = df_txt %>% mutate(ID = c(1:nrow(df_txt)))

df_txt = df_txt[,c("ID","boilerplate","label")]
# df_txt$label = df_txt$label == "1"

outcome = df_txt %>% select(ID, label)

word_list = df_txt %>% unnest_tokens(word, `boilerplate`) %>%
  group_by(ID) %>% 
  distinct() %>%
  ungroup() %>%    
  anti_join(stop_words) %>%
  count(word, sort = TRUE) %>%
  filter(n >= 50) %>%
  pull(word)

# ratio = df_txt %>% 
#   unnest_tokens(word, `boilerplate`) %>%
#   group_by(ID) %>% 
#   distinct() %>%
#   ungroup() %>%
#   filter(word %in% word_list) %>%
#   count(label, word) %>%
#   filter(n >= 100) %>%
#   pivot_wider(names_from = label, values_from = n) %>%
#   replace(is.na(.), 1) %>%
#   mutate(ratio = `1`/`0`) %>%
#   arrange(desc(ratio)) %>%
#   select(word,ratio)
# 
# most_informative_words = combine(tail(ratio$word,100),head(ratio$word,100))

bag_of_words = df_txt %>% unnest_tokens(word, `boilerplate`) %>%
  group_by(ID) %>% 
  distinct() %>%
  ungroup() %>%  
  count(ID, word) %>%
  filter(word %in% word_list) %>%
  pivot_wider(id_cols = ID, names_from = word, values_from = n) %>%
  right_join(outcome, by="ID") %>%
  replace(is.na(.), 0)  %>%
  select(-ID) 

df_nb = read.csv('~/Desktop/Masters/Classes/Machine Learning/Projects/Evergreen - stumbleupon/Data/train.csv')
df_nb = cbind(df_nb$label,bag_of_words)
colnames(df_nb)[1] = "label"

everyColumn = colnames(df_nb)
df_nb[everyColumn] = lapply(df_nb[everyColumn], factor)

idx = sample(nrow(df_nb),0.75*nrow(df_nb))
train_nb = df_nb[ idx,]
test_nb = df_nb[-idx,]
library(e1071)
model_nb = naiveBayes(label ~ ., data = train_nb)
pred_nb = predict(model_nb, test_nb)

errorRate_nb = sum(pred_nb != test_nb$label)/nrow(test_nb)

########################### Stacking with Random Forest ###############################

#Import
df<-read.csv("~/Desktop/Masters/Classes/Machine Learning/Projects/Evergreen - stumbleupon/Data/train.csv")

#Manage
df$alchemy_category<-as.factor(df$alchemy_category)
df$is_news<-as.factor(df$is_news)
df$lengthyLinkDomain<-as.factor(df$lengthyLinkDomain)
df$news_front_page<-as.factor(df$news_front_page)
df$hasDomainLink<-as.factor(df$hasDomainLink)
df$label<-as.factor(df$label)
df$framebased<-as.factor(df$framebased)

df$url<-NULL
df$urlid<-NULL
df$boilerplate<-NULL
df$alchemy_category_score<-NULL

##Step 1
#input1: Logistic
pred_glm_full<-predict(model_glm,df_glm)
#input 2: Tree
pred_tree_full<-predict(model_tree,df_tree)
#input 3: Naive bayes
pred_nb_full<-predict(model_nb, df_nb)

##Step 2
df_stack_og<-cbind(df, pred_glm_full, pred_tree_full, pred_nb_full)

#RANDOM FOREST DOES NOT LIKE WHEN COLUMNS START WITH NUMBERS SO WE HAVE
#TO ADD A RANDOM LETTER (A) TO THE BEGINNING OF EACH COLUMN

df_stack<-df_stack_og
colnames(df_stack)[25] = "A0"
colnames(df_stack)[26] = "A1"


##Step 3
N_stack<- nrow(df_stack) 
training_size_stack<-round(N_stack*0.6) 
train_cases_stack<-sample(N_stack, training_size_stack) 
train_stack<-df_stack[train_cases_stack,]
test_stack<-df_stack[-train_cases_stack,]

#Stacked Model
#Model
library(randomForest)
model_stack<-randomForest(label ~., data = train_stack, ntree=750)

#PREDICT
pred_stack<-predict(model_stack, test_stack)

#EVALUATE
error_stack<-sum(pred_stack!= test_stack$label)/nrow(test_stack)
error_bench_stack=benchmarkErrorRate(train_stack$label, test_stack$label)
summary(model_stack)

observations_stacked=test_stack$label
table(pred_stack,observations_stacked)

varImpPlot(model_stack)
varImp(model_stack)
