rm(list = ls())

add_col <- function(df, co_l) {
  # NAs to concatenate unequal length cols   
  length(co_l) <- nrow(df)
  cbind(df, co_l)
}

## diabetes and boston already have splits

## airfoil
set.seed(42)
airfoil <- read.table("application/benchmarks/data/airfoil/airfoil_self_noise.dat")
air_train <- sample(1:nrow(airfoil), round(nrow(airfoil)*0.75))
air_test <- setdiff(1:nrow(airfoil), air_train)

## forest fire
set.seed(42)
ff <- read.csv("application/benchmarks/data/forestfires/forestfires.csv")
fire_train <- sample(1:nrow(ff), round(nrow(ff)*0.75))
fire_test <- setdiff(1:nrow(ff), fire_train)

d <- data.frame(air_train = air_train)
d <- add_col(d, air_test)
d <- add_col(d, fire_train)
d <- add_col(d, fire_test)
colnames(d) <- c("air_train","air_test","fire_train","fire_test")

write.csv(d, "application/benchmarks/data/bench_split.csv")
