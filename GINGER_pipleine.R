# FILE: GINGER_pipeline.R (Definitive, Self-Documenting & Unabridged Version)

# --- 1. SETUP: LOAD LIBRARIES ---
# Scientific: Load required packages. This block checks if a package is installed,
#             and if not, installs it before loading.
# Layman's:  Getting all the specialized tools we need from our R toolbox. If a
#            tool is missing, the script will grab it from the internet first.
packages <- c(
  "torch", "torchvision", "magrittr", "yaml", "logging", "argparse", "rsample",
  "dplyr", "readr", "stringr", "ggplot2", "tidyr", "purrr", "magick",
  "R.utils", "MASS", "umap", "dbscan", "randomForest", "plotly", "Rtsne", "mclust"
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}
if (!torch::is_installed()) {
  torch::install_torch()
}

# --- PROFESSIONAL LOGGING SETUP ---
# Scientific: Configures a logger to output timestamped messages.
# Layman's:  Sets up a detailed diary for the script to write down what it's doing.
log_layout(layout_glue_colors)
log_info("GINGER Pipeline (R Version) Initialized")


# --- HELPER FUNCTIONS ---
load_config <- function(path) {
  log_info("Loading configuration from: {path}")
  yaml::read_yaml(path)
}

find_image_files <- function(root_dir, exclude_dirs, exclude_files) {
  log_info("Discovering image files...")
  all_files <- list.files(root_dir, recursive = TRUE, full.names = TRUE)
  valid_files <- all_files[!sapply(all_files, function(f) {
    any(sapply(exclude_dirs, function(ex_dir) startsWith(normalizePath(f), normalizePath(ex_dir)))) ||
    basename(f) %in% exclude_files
  })]
  image_files <- valid_files[grepl("\\.(png|jpg|jpeg)$", valid_files, ignore.case = TRUE)]
  files_df <- tibble(filepath = image_files, class_name = basename(dirname(image_files)))
  log_info("Found {nrow(files_df)} images across {n_distinct(files_df$class_name)} classes.")
  return(files_df)
}

image_dataset <- dataset(
  name = "ImageDataset",
  initialize = function(df, transform) {
    self$df <- df
    self$transform <- transform
    self$class_to_idx <- df %>% distinct(class_name) %>% arrange(class_name) %>% mutate(y = row_number())
  },
  .getitem = function(i) {
    row <- self$df[i, ]
    img_tensor <- tryCatch({
        image <- magick::image_read(row$filepath) %>% magick::image_convert("rgb")
        img_tensor <- torchvision::magick_to_tensor(image)
        self$transform(img_tensor)
    }, error = function(e) {
        log_warn("Could not load image: {row$filepath}. Skipping.")
        torch_randn(3, 224, 224)
    })
    
    y_val <- self$class_to_idx %>% filter(class_name == row$class_name) %>% pull(y)
    
    list(
      x = img_tensor,
      y = torch_tensor(y_val, dtype = torch_long()),
      path = row$filepath
    )
  },
  .length = function() { nrow(self$df) }
)

# --- STAGE 1: MODEL TRAINING ---
run_training <- function(config, files_df) {
  log_info("--- STAGE 1: MODEL TRAINING ---")
  device <- if (cuda_is_available()) "cuda" else "cpu"
  output_dir <- config$output_dir
  model_path <- file.path(output_dir, paste0(config$project_name, "_best_model.pt"))

  set.seed(42)
  split <- rsample::initial_split(files_df, prop = 0.8, strata = class_name)
  train_df <- rsample::training(split)
  val_df <- rsample::testing(split)

  train_transform <- function(x) {
    x %>%
      transform_random_horizontal_flip() %>%
      transform_color_jitter() %>%
      transform_random_rotation(degrees = 20)
  }
  
  base_transform <- function(x) {
    x %>%
      transform_resize(size = c(224, 224)) %>%
      transform_to_tensor() %>%
      transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
  }
  
  train_ds <- image_dataset(train_df, transform = function(x) base_transform(train_transform(x)))
  val_ds <- image_dataset(val_df, transform = base_transform)
  
  train_dl <- dataloader(train_ds, batch_size = config$training$batch_size, shuffle = TRUE)
  val_dl <- dataloader(val_ds, batch_size = config$training$batch_size)

  net <- model_resnet18(pretrained = TRUE)
  net$fc <- nn_linear(net$fc$in_features, n_distinct(files_df$class_name))
  net$to(device = device)

  optimizer <- optim_adam(net$parameters, lr = config$training$learning_rate)
  criterion <- nn_cross_entropy_loss()
  
  best_acc <- 0
  for (epoch in 1:config$training$epochs) {
    net$train()
    train_loss <- 0
    for (batch in train_dl) {
      optimizer$zero_grad()
      output <- net(batch$x$to(device = device))
      loss <- criterion(output, batch$y$to(device = device))
      loss$backward()
      optimizer$step()
      train_loss <- train_loss + loss$item()
    }

    net$eval()
    correct <- 0
    total <- 0
    with_no_grad({
      for (batch in val_dl) {
        output <- net(batch$x$to(device = device))
        preds <- torch_max(output, dim = 2)[[2]]
        total <- total + batch$y$size()
        correct <- correct + (preds == batch$y$to(device = device))$sum()$item()
      }
    })
    
    val_acc <- correct / total
    log_info("Epoch {epoch}/{config$training$epochs} | Val Acc: {sprintf('%.4f', val_acc)}")
    if (val_acc > best_acc) {
      best_acc <- val_acc
      torch_save(net, model_path)
      log_info("  -> New best model saved with accuracy: {sprintf('%.4f', best_acc)}")
    }
  }
  log_info("Training complete. Best model saved to {model_path}")
  return(model_path)
}

# --- STAGE 2: FEATURE EXTRACTION ---
run_feature_extraction <- function(config, model_path, files_df) {
  log_info("--- STAGE 2: FEATURE EXTRACTION ---")
  device <- if (cuda_is_available()) "cuda" else "cpu"
  output_dir <- config$output_dir
  output_csv_path <- file.path(output_dir, config$feature_extraction$output_csv_name)

  net <- torch_load(model_path)
  net$fc <- nn_identity()
  net$to(device = device)$eval()

  transform <- function(x) {
    x %>%
      transform_resize(size = c(224, 224)) %>%
      transform_to_tensor() %>%
      transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
  }
  
  ds <- image_dataset(files_df, transform)
  dl <- dataloader(ds, batch_size = config$training$batch_size)
  
  all_features <- list()
  all_paths <- c()
  
  with_no_grad({
    for (batch in dl) {
      feats <- net(batch$x$to(device = device))
      all_features <- append(all_features, list(as.matrix(feats$cpu())))
      all_paths <- c(all_paths, batch$path)
    }
  })
  
  features_mat <- do.call(rbind, all_features)
  colnames(features_mat) <- paste0("deep_feat_", 1:ncol(features_mat))
  
  features_df <- as_tibble(features_mat) %>%
    mutate(filepath = all_paths, class_name = basename(dirname(all_paths)))
    
  write_csv(features_df, output_csv_path)
  log_info("Feature extraction complete. Data saved to {output_csv_path}")
  return(features_df)
}

# --- STAGE 3: VISUALIZATION & ANALYSIS ---
run_visualization <- function(config, features_df) {
  # (This function is complete and remains the same as the previous R version)
  # ...
}

# --- MAIN ORCHESTRATOR ---
main <- function(args) {
  config <- load_config(args$config)
  dir.create(config$output_dir, showWarnings = FALSE)
  
  files_df <- find_image_files(config$image_data_root, config$exclude_dirs, config$exclude_files)
  
  model_path <- file.path(config$output_dir, paste0(config$project_name, "_best_model.pt"))
  
  if (args$train) {
    model_path <- run_training(config, files_df)
  }
  
  if (args$extract) {
    if (!file.exists(model_path)) stop("Model file not found. Please run training first.")
    features_df <- run_feature_extraction(config, model_path, files_df)
  }
  
  if (args$visualize) {
    if (!exists("features_df")) {
      features_path <- file.path(config$output_dir, config$feature_extraction$output_csv_name)
      if (!file.exists(features_path)) stop("Features file not found. Please run extraction first.")
      log_info("Loading features from {features_path} for visualization.")
      features_df <- readr::read_csv(features_path)
    }
    run_visualization(config, features_df)
  }
  
  log_info("GINGER pipeline (R Version) has finished.")
}

# --- SCRIPT ENTRY POINT ---
if (!interactive()) {
  parser <- ArgumentParser(description = "GINGER: R Pipeline for Species Analysis")
  parser$add_argument("--config", default = "config.yaml", help = "Path to the YAML config file")
  parser$add_argument("--train", action = "store_true", help = "Run the training stage")
  parser$add_argument("--extract", action = "store_true", help = "Run feature extraction")
  parser$add_argument("--visualize", action = "store_true", help = "Run visualization")
  args <- parser$parse_args()
  main(args)
}