library(devtools)
library(parallel)

##########

nr_words = 10000
embedding_size = 100
maxlen = 100
order_bsp = 25

if (!file.exists("data_splitted.RDS")){
  
  library(keras)
  library(tidyverse)
  library(jsonlite)
  library(lubridate)
  library(ggplot2)
  library(ggsci)
  library(tidytext)
  library(tm)
  library(data.table)
  theme_set(theme_bw())
  
  # read in data
  movies <- read_csv("movies.csv")
  
  # get genres
  movies <- movies %>%
    filter(original_language == "en" & status == "Released") %>% 
    # filter(nchar(genres)>2) %>%
    mutate(genres = lapply(genres, function(x) fromJSON(x)$name)) %>%  
    select(genres, budget, overview, popularity, 
           production_countries, release_date, 
           revenue, runtime, vote_average, 
           vote_count) %>% 
    filter(vote_count > 0) %>% 
    mutate(release_date = as.numeric(as.Date("2020-01-01") - as.Date(release_date)))
  
  genres <- movies %>%  unnest(genres, .name_repair = "unique") %>% select(genres) 
  table(genres)
  
  movies <- movies[unlist(sapply(movies$genres, function(x) 
    length(x) > 1 | (!("TV Movie" %in% x) & !("Foreign" %in% x)))),]
  
  
  # vote_average
  (
    ggplot(movies %>%  unnest(genres, .name_repair = "unique") %>% 
             filter(!genres %in% c(NA, "TV Movie", "Foreign")), 
           aes(x = log(revenue), fill = genres)) + 
      geom_density(alpha = 0.3) + xlab("log. revenue") +
      theme_bw() + 
      theme(
        axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        legend.title = element_blank(),
        text = element_text(size = 14),
        legend.position = "bottom") + 
      scale_color_jco() +
      guides(fill = guide_legend(nrow=3, byrow=TRUE))) %>% 
    ggsave(filename = "log_revenue.pdf", width = 6, height = 4.5)
  
  all_genres <- sort(unique(unlist(movies$genres)))
  genres_wide <- t(sapply(movies$genres,
                          function(x) 
                            colSums(model.matrix(~ -1 + genre, 
                                                 data = data.frame(genre=factor(x, levels = all_genres))))
  ))
  
  colnames(genres_wide) <- gsub(" ", "_", colnames(genres_wide))
  
  movies <- cbind(movies %>% select(-genres), genres_wide)                      
  
  
  # init tokenizer
  tokenizer <- text_tokenizer(num_words = nr_words)
  
  # remove stopwords
  data("stop_words")
  stopwords_regex = paste(c(stopwords('en'), stop_words$word), 
                          collapse = '\\b|\\b')
  stopwords_regex = paste0('\\b', stopwords_regex, '\\b')
  movies$overview = tolower(movies$overview)
  movies$overview = stringr::str_replace_all(movies$overview, stopwords_regex, '')
  movies$overview = gsub('[[:punct:] ]+',' ', movies$overview)
  
  saveRDS(movies$overview, file="mov_ov.RDS")
  
  tokenizer %>% fit_text_tokenizer(movies$overview)
  
  # text to sequence
  text_seqs <- texts_to_sequences(tokenizer, movies$overview)
  
  # pad text sequences
  text_padded <- text_seqs %>%
    pad_sequences(maxlen = maxlen)
  
  # save words for later
  words <- data_frame(
    word = names(tokenizer$word_index), 
    id = as.integer(unlist(tokenizer$word_index))
  )
  
  words <- words %>%
    filter(id <= tokenizer$num_words) %>%
    arrange(id)
  
  saveRDS(words, file="words.RDS")
  rm(words)
  gc()
  
  # text sequences as list of one array
  text_embd <- list(texts = array(text_padded, dim=c(NROW(movies), maxlen)) )
  
  # create input list
  data <- append(movies, text_embd) 
  
  rm(movies, text_embd)
  gc()
  
  repetitions <- 10
  
  data_list <- list()
  
  # repeat analysis 10 times
  for(repl in 1:repetitions){
    
    set.seed(41 + repl)
    
    train_ind <- sample(1:NROW(data$runtime), round(0.8*NROW(data$runtime)))
    test_ind <- setdiff(1:NROW(data$runtime), train_ind)
    
    train <- lapply(data, function(x) if(length(dim(x))==2) x[train_ind,] else x[train_ind])
    test <- lapply(data, function(x) if(length(dim(x))==2) x[test_ind,] else x[test_ind])
    
    data_list[[repl]] <- list(train = train, test = test)
    
  }
  
  saveRDS(data_list, "data_splitted.RDS")
  
  
}else{
  
  data_list <- readRDS("data_splitted.RDS")
  
  
}

res <- list()
res_list <- list()
epochs <- 10000
nrCores <- length(data_list)

# repeat analysis 10 times
res_list <- mclapply(1:length(data_list), function(repl){
  
  library(lubridate)
  
  # load software
  load_all("~/deepregression_master/deepregression/R") 
  
  # reinitialize tokenizer
  tokenizer <- text_tokenizer(num_words = nr_words)
  tokenizer %>% fit_text_tokenizer(readRDS("mov_ov.RDS"))
  
  train <- data_list[[repl]]$train
  test <- data_list[[repl]]$test
  
  # optimizer
  optimizer <- optimizer_adadelta(lr = 0.1)
  
  ##########################################################################
  ###################### only structured ###################################
  
  form_lists <- list(
    shift = ~ 1 + 
      s(budget) + 
      s(popularity) + 
      s(release_date) + 
      s(runtime) + 
      genreAction + 
      genreAdventure + 
      genreAnimation + 
      genreComedy + 
      genreCrime + 
      genreDocumentary +
      genreDrama + 
      genreFamily + 
      genreFantasy + 
      genreForeign + 
      genreHistory + 
      genreHorror + 
      genreMusic + 
      genreMystery + 
      genreRomance + 
      genreScience_Fiction + 
      genreThriller + 
      genreTV_Movie + 
      genreWar + 
      genreWestern,
    interaction = ~ -1 + 
      s(budget, bs = "bs") +
      s(popularity, bs = "bs") +
      s(release_date, bs = "bs") +
      s(runtime, bs = "bs") + 
      genreAction + 
      genreAdventure + 
      genreAnimation + 
      genreComedy + 
      genreCrime + 
      genreDocumentary +
      genreDrama + 
      genreFamily + 
      genreFantasy + 
      genreForeign + 
      genreHistory + 
      genreHorror + 
      genreMusic + 
      genreMystery + 
      genreRomance + 
      genreScience_Fiction + 
      genreThriller + 
      genreTV_Movie + 
      genreWar + 
      genreWestern
  )
  
  mod <- deepregression(y = train$revenue, 
                        list_of_formulae = form_lists, 
                        list_of_deep_models = list(NULL), 
                        family = "transformation_model",
                        order_bsp = order_bsp, 
                        zero_constraint_for_smooths = FALSE,
                        # addconst_interaction = 0,
                        data = train,
                        df = 9, 
                        sp_scale = 256,
                        optimizer = optimizer
  )
  
  mod %>% fit(epochs = epochs, 
              validation_split = 0.2,
              batch_size = 256,
              view_metrics = FALSE, 
              verbose = T,
              callbacks = 
                list(
                  callback_early_stopping(patience = 60),
                  callback_reduce_lr_on_plateau(patience = 20)
                )
  )
  
  (
    pr_log_scores <- log_score(mod, data = test, this_y = test$revenue, 
                               summary_fun = sum)
  )/888
  
  shift <- get_shift(mod)
  theta <- get_theta(mod)
  plotData_shift <- plot(mod, plot = F, which_param = 1)
  plotData_theta <- plot(mod, plot = F, which_param = 2)
  
  res[[1]] <- list(pls = pr_log_scores,
                   shift = shift,
                   theta = theta,
                   plotData_shift = plotData_shift,
                   plotData_theta = plotData_theta)
  
  str_w1 <- mod$model$get_layer(name="structured_nonlinear_1")$get_weights()
  str_w2 <- mod$model$get_layer(name="constraint_mono_layer_multi")$get_weights()
  
  ##########################################################################
  ###################### embd in shift #####################################
  
  optimizer <- optimizer_adam()
  
  # modeling
  embd_mod <- function(x) x %>%
    layer_embedding(input_dim = tokenizer$num_words,
                    output_dim = embedding_size) %>%
    # layer_lstm(units = 20, return_sequences = TRUE) %>% 
    # layer_dropout(rate = 0.3) %>% 
    # layer_lstm(units = 5) %>% 
    layer_flatten() %>% 
    # layer_dropout(rate = 0.3) %>% 
    layer_dense(1)
  
  form_lists <- list(
    shift = ~ 1 + 
      s(budget) + 
      s(popularity) + 
      s(release_date) + 
      s(runtime) + 
      genreAction + 
      genreAdventure + 
      genreAnimation + 
      genreComedy + 
      genreCrime + 
      genreDocumentary +
      genreDrama + 
      genreFamily + 
      genreFantasy + 
      genreForeign + 
      genreHistory + 
      genreHorror + 
      genreMusic + 
      genreMystery + 
      genreRomance + 
      genreScience_Fiction + 
      genreThriller + 
      genreTV_Movie + 
      genreWar + 
      genreWestern +
      embd_mod(texts),
    interaction = ~ -1 + 
      s(budget, bs = "bs") +
      s(popularity, bs = "bs") +
      s(release_date, bs = "bs") +
      s(runtime, bs = "bs") + 
      genreAction + 
      genreAdventure + 
      genreAnimation + 
      genreComedy + 
      genreCrime + 
      genreDocumentary +
      genreDrama + 
      genreFamily + 
      genreFantasy + 
      genreForeign + 
      genreHistory + 
      genreHorror + 
      genreMusic + 
      genreMystery + 
      genreRomance + 
      genreScience_Fiction + 
      genreThriller + 
      genreTV_Movie + 
      genreWar + 
      genreWestern
  )
  
  mod <- deepregression(y = train$revenue, 
                        list_of_formulae = form_lists, 
                        list_of_deep_models = list(embd_mod = embd_mod), 
                        family = "transformation_model",
                        order_bsp = order_bsp,
                        zero_constraint_for_smooths = FALSE,
                        data = train,
                        df = 9, 
                        sp_scale = 256,
                        optimizer = optimizer
  )
  
  mod$model$get_layer(name="structured_nonlinear_1")$set_weights(str_w1)
  
  mod$model$get_layer(name="constraint_mono_layer_multi")$set_weights(str_w2)
  
  
  mod %>% fit(epochs = epochs, 
              validation_split = 0.2,
              batch_size = 256,
              view_metrics = FALSE, 
              verbose = T,
              callbacks = 
                list(
                  callback_early_stopping(patience = 60),
                  callback_reduce_lr_on_plateau(patience = 20)
                )
  )
  
  (pr_log_scores <- log_score(mod, data = test, this_y = test$revenue, 
                              summary_fun = sum)
  )/888
  
  shift <- get_shift(mod)
  theta <- get_theta(mod)
  plotData_shift <- plot(mod, plot = F, which_param = 1)
  plotData_theta <- plot(mod, plot = F, which_param = 2)
  
  res[[2]] <- list(pls = pr_log_scores,
                   shift = shift,
                   theta = theta,
                   plotData_shift = plotData_shift,
                   plotData_theta = plotData_theta)
  
  if(FALSE){
    
    # save also pdfs
    
    trf <- mod %>% predict(test)
    gr_y <- seq(min(test$revenue),
                max(test$revenue), l = 888)
    pdf_y <- trf(gr_y, 
                 type = "pdf", grid = T)
    pdf_y[pdf_y<0] <- 0
    
    par(mfrow = c(4,5))
    for(i in 1:20){
      
      genre_i_ind <- test[[9+i]]==1
      matplot(log(gr_y), pdf_y[,genre_i_ind], type="l", col = rgb(0,0,0,0.2),
              ylab = "density", xlab = "Log. Revenue",
              lty = 1, main = gsub("genre", "", names(test)[i+10]))
      
    }
    
    words <- readRDS("words.RDS")
    
    search_words <- c("intelligent", "sexy", "brutal")[c(3,2,1)]
    
    ind1 <- apply(test$texts, 1, 
                  function(seq) (words %>% 
                                   filter(word==search_words[1]) %>% 
                                   select(id) %>% 
                                   pull()) %in% seq)
    
    ind2 <- apply(test$texts, 1, 
                  function(seq) (words %>% 
                                   filter(word==search_words[2]) %>% 
                                   select(id) %>% 
                                   pull()) %in% seq)
    
    ind3 <- apply(test$texts, 1, 
                  function(seq) (words %>% 
                                   filter(word==search_words[3]) %>% 
                                   select(id) %>% 
                                   pull()) %in% seq)
    
    lwd = 3
    
    library("colorspace")
    
    col <- qualitative_hcl(3, c(240, 0), l = 60)
    
    saveRDS(list(gr_y=gr_y, pdf_y=pdf_y, ind1=ind1, ind2=ind2, ind3=ind3),
            file = "data_plot_densities.RDS")
    
    par(mfrow = c(1,1), mar = c(5.1, 5.1, 4.1, 2.1))
    matplot(log(gr_y), pdf_y[,ind1]*(1e8), type="l", col = col[1],
            ylab = expression(paste("density ", {}%.%10^-8)), 
            xlab = "log. revenue", bty="n",
            lty = 2, lwd=lwd, xlim = c(17,21), ylim = c(0,4))
    matplot(log(gr_y), pdf_y[,ind2]*(1e8), type="l", col = col[3],
            lty = 3, lwd=lwd, add = TRUE)
    matplot(log(gr_y), pdf_y[,ind3]*(1e8), type="l", col = col[2],
            lty = 4, lwd=lwd, add = TRUE)
    legend(x = 20, y = 4, lwd=lwd, lty = 2:4, col = col[c(1,3,2)], search_words)
    
    nr_res <- sum(ind1+ind2+ind3)
    
    plot_data <- data.frame(pdf_y = c(c(pdf_y[,ind1]), c(pdf_y[,ind2]), c(pdf_y[,ind3])),
                            obs = rep(1:nr_res, each=888),
                            gr_y = rep(log(gr_y),nr_res),
                            ind = rep(search_words, 888*sapply(list(ind1,ind2,ind3),sum)))
    
    saveRDS(plot_data, file = "plot_data.RDS")
    
    library(ggsci)
    
    (
      ggplot(plot_data, 
             aes(x=gr_y, y=pdf_y*1e8, group=obs, colour=ind, linetype=ind)) +
        geom_line(size=1.5) + xlim(17,20.5) + scale_color_jco() +
        ylab(expression(paste("density ", {}%.%10^-8))) + 
        xlab("log. revenue") + 
        theme(axis.line = element_line(colour = "black"),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.border = element_blank(),
              panel.background = element_blank(),
              legend.title = element_blank(),
              text = element_text(size = 14)) ) %>% 
      ggsave(filename = "words_density.pdf", width = 6, height = 4.5)
    
    
  }
  
  ##########################################################################
  ###################### embd in theta #####################################
  
  # modeling
  embd_mod <- function(x) x %>%
    layer_embedding(input_dim = tokenizer$num_words,
                    output_dim = embedding_size) %>%
    # layer_lstm(units = 20, return_sequences = TRUE) %>% 
    # layer_dropout(rate = 0.3) %>% 
    # layer_lstm(units = 5) %>% 
    layer_flatten() %>% 
    # layer_dropout(rate = 0.3) %>% 
    layer_dense(1, activation = "relu")
  
  form_lists <- list(
    shift = ~ 1 + 
      s(budget) + 
      s(popularity) + 
      s(release_date) + 
      s(runtime) + 
      genreAction + 
      genreAdventure + 
      genreAnimation + 
      genreComedy + 
      genreCrime + 
      genreDocumentary +
      genreDrama + 
      genreFamily + 
      genreFantasy + 
      genreForeign + 
      genreHistory + 
      genreHorror + 
      genreMusic + 
      genreMystery + 
      genreRomance + 
      genreScience_Fiction + 
      genreThriller + 
      genreTV_Movie + 
      genreWar + 
      genreWestern,
    interaction = ~ -1 + 
      s(budget, bs = "bs") +
      s(popularity, bs = "bs") +
      s(release_date, bs = "bs") +
      s(runtime, bs = "bs") + 
      genreAction + 
      genreAdventure + 
      genreAnimation + 
      genreComedy + 
      genreCrime + 
      genreDocumentary +
      genreDrama + 
      genreFamily + 
      genreFantasy + 
      genreForeign + 
      genreHistory + 
      genreHorror + 
      genreMusic + 
      genreMystery + 
      genreRomance + 
      genreScience_Fiction + 
      genreThriller + 
      genreTV_Movie + 
      genreWar + 
      genreWestern +
      embd_mod(texts)
  )
  
  mod <- deepregression(y = train$revenue, 
                        list_of_formulae = form_lists, 
                        list_of_deep_models = list(embd_mod = embd_mod), 
                        family = "transformation_model",
                        order_bsp = order_bsp,
                        zero_constraint_for_smooths = FALSE,
                        data = train,
                        df = 9, 
                        sp_scale = 256,
                        optimizer = optimizer
  )
  
  mod$model$get_layer(name="structured_nonlinear_1")$set_weights(str_w1)
  
  mod$model$get_layer(name="constraint_mono_layer_multi")$set_weights(list(
    rbind(
      str_w2[[1]],
      matrix(rnorm(13*2),ncol=1))
  )
  )
  
  mod %>% fit(epochs = epochs, 
              validation_split = 0.2,
              batch_size = 256,
              view_metrics = FALSE, 
              verbose = T,
              callbacks = 
                list(
                  callback_early_stopping(patience = 60),
                  callback_reduce_lr_on_plateau(patience = 20)
                )
  )
  
  (
    pr_log_scores <- log_score(mod, data = test, this_y = test$revenue, 
                               summary_fun = sum)
  )/888
  
  shift <- get_shift(mod)
  theta <- get_theta(mod)
  plotData_shift <- plot(mod, plot = F, which_param = 1)
  plotData_theta <- plot(mod, plot = F, which_param = 2)
  
  res[[3]] <- list(pls = pr_log_scores,
                   shift = shift,
                   theta = theta,
                   plotData_shift = plotData_shift,
                   plotData_theta = plotData_theta)
  
  
  ##########################################################################
  ###################### embd in both ######################################
  
  # optimizer <- optimizer_adam()
  
  # modeling
  embd_mod <- function(x) x %>%
    layer_embedding(input_dim = tokenizer$num_words,
                    output_dim = embedding_size) %>%
    # layer_lambda(f = function(x) k_mean(x, axis = 2)) %>%
    # layer_dense(10) %>% 
    layer_flatten() %>% 
    layer_dense(1+1, activation = "relu")
  
  form_lists <- list(
    shift = ~ 1 + 
      s(budget) + 
      s(popularity) + 
      s(release_date) + 
      s(runtime) +
      genreAction + 
      genreAdventure + 
      genreAnimation + 
      genreComedy + 
      genreCrime + 
      genreDocumentary +
      genreDrama + 
      genreFamily + 
      genreFantasy + 
      genreForeign + 
      genreHistory + 
      genreHorror + 
      genreMusic + 
      genreMystery + 
      genreRomance + 
      genreScience_Fiction + 
      genreThriller + 
      genreTV_Movie + 
      genreWar + 
      genreWestern,
    interaction = ~ -1 + 
      s(budget, bs = "bs") +
      s(popularity, bs = "bs") +
      s(release_date, bs = "bs") +
      s(runtime, bs = "bs") + 
      genreAction + 
      genreAdventure + 
      genreAnimation + 
      genreComedy + 
      genreCrime + 
      genreDocumentary +
      genreDrama + 
      genreFamily + 
      genreFantasy + 
      genreForeign + 
      genreHistory + 
      genreHorror + 
      genreMusic + 
      genreMystery + 
      genreRomance + 
      genreScience_Fiction + 
      genreThriller + 
      genreTV_Movie + 
      genreWar + 
      genreWestern
  )
  
  tt_list <- list( ~ + embd_mod(texts))[rep(1,2)]
  
  
  mod <- deepregression(y = train$revenue, 
                        list_of_formulae = form_lists, 
                        list_of_deep_models = list(embd_mod = embd_mod), 
                        train_together = tt_list,
                        family = "transformation_model",
                        order_bsp = order_bsp,
                        zero_constraint_for_smooths = FALSE,
                        data = train,
                        optimizer = optimizer,
                        df = list(9,9,0), 
                        sp_scale = 256,
                        split_between_shift_and_theta = c(1,1)
  )
  
  mod$model$get_layer(name="structured_nonlinear_1")$set_weights(str_w1)
  
  mod$model$get_layer(name="constraint_mono_layer_multi")$set_weights(
    list(
      rbind(
      str_w2[[1]],matrix(rnorm(13*2),ncol=1))
      ))
  
  mod %>% fit(epochs = epochs, 
              validation_split = 0.2,
              batch_size = 256,
              view_metrics = FALSE, 
              verbose = T,
              callbacks = 
                list(
                  callback_early_stopping(patience = 60),
                  callback_reduce_lr_on_plateau(patience = 20)
                )
  )
  
  (
    pr_log_scores <- log_score(mod, data = test, this_y = test$revenue, 
                               summary_fun = sum)
  )/888
  
  shift <- get_shift(mod)
  theta <- get_theta(mod)
  plotData_shift <- plot(mod, plot = F, which_param = 1)
  plotData_theta <- plot(mod, plot = F, which_param = 2)
  
  res[[4]] <- list(pls = pr_log_scores,
                   shift = shift,
                   theta = theta,
                   plotData_shift = plotData_shift,
                   plotData_theta = plotData_theta)
  
  
  return(res)
  
}, mc.cores = nrCores)

saveRDS(res_list, file = "res_pen.RDS")

if(FALSE)
{
  
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(ggsci)
  library(knitr)
  theme_set(theme_bw())
  
  # Analyze results
  
  res <- readRDS("res_pen.RDS")
  model_names <- c("Structured", "Deep Shift", "Deep Interaction", "Deep Combination")
  
  #### predictions
  
  # predictive log-scores
  pls <- t(sapply(res, function(x) sapply(x, "[[", "pls")))/888 # 888 = #test data
  data.frame(models = model_names,
             pls = apply(pls, 2, function(x) paste0(signif(mean(x),4), " (",
                                                    signif(stats::sd(x),4), ")"))) %>% kable("latex")
  
  #### compare smooth effects
  
  ##### shift
  
  res_shift <- do.call("rbind", lapply(1:length(res), function(i){
    repetition_i <- res[[i]]
    ri <- do.call("rbind", lapply(1:length(repetition_i), function(j){
      
      model_j <- repetition_i[[j]]
      res_model_i <- do.call("rbind", lapply(model_j$plotData_shift, 
                                             function(feature_k)
                                               return(
                                                 data.frame(feature = feature_k$org_feature_name,
                                                            value = feature_k$value[,1],
                                                            partial_effect = feature_k$partial_effects[,1])
                                               )))
      res_model_i$model <-  model_names[j]
      return(res_model_i)
    }))
    ri$rep <- i
    return(ri)
  }))
  
  res_shift$model <- factor(res_shift$model, 
                            levels = model_names)
  
  res_shift$value <- log(res_shift$value, base = 10)
  
  (ggplot(res_shift, aes(x = value, y = partial_effect, group = rep)) + 
      geom_line(aes(alpha = 0.4)) + facet_grid(model ~ feature, scales = "free") + 
      xlab("value") + ylab("partial effect") + guides(alpha=FALSE) ) 
  %>% 
    ggsave(filename = "splines_shift.pdf", width = 12, height = 10)
  
  levels(res_shift$feature)[3] <- "release date"
  
  (ggplot(res_shift  %>% filter(rep==4), 
          aes(x = exp(value), y = partial_effect, colour = model, linetype=model)) + 
      geom_line(size=1.3, alpha = 0.8) + facet_wrap( ~ feature, scales = "free", nrow=2) + 
      xlab("value (log-scale)") + ylab("partial effect") + guides(alpha=FALSE) + 
      theme(text = element_text(size = 14), 
            legend.position = "bottom",
            legend.title = element_blank()) + scale_color_jco() + 
      guides(colour = guide_legend(nrow=2)))  %>% 
    ggsave(filename = "splines_shift.pdf", width = 5, height = 5)
  
  ##### theta (for one BSP coefficient)
  
  bsp_coef_nr <- 3
  
  res_ia <- do.call("rbind", lapply(1:length(res), function(i){
    repetition_i <- res[[i]]
    ri <- do.call("rbind", lapply(1:length(repetition_i), function(j){
      
      model_j <- repetition_i[[j]]
      res_model_i <- do.call("rbind", 
                             lapply(model_j$plotData_theta, 
                                    function(feature_k)
                                      return(
                                        data.frame(feature = feature_k$org_feature_name,
                                                   value = feature_k$value[,1],
                                                   partial_effect = feature_k$partial_effects[,bsp_coef_nr])
                                      )))
      res_model_i$model <-  model_names[j]
      return(res_model_i)
    }))
    ri$rep <- i
    return(ri)
  }))
  
  res_ia$model <- factor(res_ia$model, 
                         levels = model_names)
  
  # res_ia$value <- log(res_ia$value, base = 10)
  
  levels(res_ia$feature)[3] <- "release date"
  
  (ggplot(res_ia, aes(x = value, y = partial_effect, group = rep)) + 
      geom_line(aes(alpha = 0.4)) + facet_grid(model ~ feature, scales = "free") + 
      xlab("value") + ylab("partial effect") + guides(alpha=FALSE)
  ) %>% 
    ggsave(filename = "splines_theta.pdf", width = 12, height = 10)
  
  (ggplot(res_ia %>% filter(rep==6), 
          aes(x = value, y = partial_effect, colour = model, linetype=model)) + 
      geom_line(size=1.3, alpha = 0.8) + facet_wrap( ~ feature, scales = "free", nrow=2) + 
      xlab("value (log-scale)") + ylab("partial effect") + guides(alpha=FALSE) + 
      theme(text = element_text(size = 14)) + scale_color_jco() ) %>% 
    ggsave(filename = "splines_theta.pdf", width = 6, height = 4)
  
  #### compare fixed effects
  
  genres <- gsub("genre", "", colnames(movies)[10:29])
  
  res_fixed_shift <- do.call("rbind", lapply(1:length(res), function(i){
    repetition_i <- res[[i]]
    ri <- do.call("rbind", lapply(1:length(repetition_i), function(j){
      
      model_j <- repetition_i[[j]]
      res_model_i <- data.frame(coef = model_j$shift[2:21],
                                genre = genres)
      res_model_i$model <-  model_names[j]
      return(res_model_i)
    }))
    ri$rep <- i
    return(ri)
  }))
  
  res_fixed_shift$model <- factor(res_fixed_shift$model, 
                                  levels = model_names)
  
  (ggplot(res_fixed_shift, aes(y = coef, x = genre, fill = model)) + 
      geom_boxplot() + scale_fill_jco() + 
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
      ylab("coefficient")) %>% 
    ggsave(filename = "coef_shift.pdf", width = 12, height = 10)
  
}
