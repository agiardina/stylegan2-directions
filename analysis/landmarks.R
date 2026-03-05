
facepp_df <- read.table("../data/all_coordinates.raw", header = TRUE)
dlib_df <- read.csv("../data/landmarks68.csv", sep = ",")

aligned_pattern <- "a_?l_?i_?g_?n_?e_?d"

facepp_aligned <- subset(facepp_df, grepl(aligned_pattern, facepp_df[[1]], ignore.case = TRUE))
dlib_aligned <- subset(dlib_df, grepl(aligned_pattern, dlib_df[[1]], ignore.case = TRUE))

if (nrow(facepp_aligned) != nrow(dlib_aligned)) {
  stop(sprintf(
    "Righe non corrispondenti dopo il filtro: facepp=%d, dlib=%d",
    nrow(facepp_aligned),
    nrow(dlib_aligned)
  ))
}

protocol_df <- read.csv("../data/protocol.csv", stringsAsFactors = FALSE, check.names = FALSE)
protocol_df$Facepp_id <- suppressWarnings(as.integer(protocol_df[["Face++"]]))
protocol_df$Dlib_id <- suppressWarnings(as.integer(protocol_df[["Dlib"]]))
protocol_df <- subset(protocol_df, !is.na(Facepp_id) & !is.na(Dlib_id))

normalize_filename <- function(x) {
  x <- sub("\\.jpg$", "", x)
  num <- sub("^([0-9]+).*", "\\1", x)
  ifelse(grepl("^[0-9]+$", num), paste0(num, "_aligned"), NA_character_)
}

facepp_aligned$Filename_norm <- normalize_filename(facepp_aligned[[1]])
dlib_aligned$Filename_norm <- normalize_filename(dlib_aligned$Filename)

joined_df <- merge(
  dlib_aligned,
  facepp_aligned,
  by.x = "Filename_norm",
  by.y = "Filename_norm",
  all = FALSE,
  sort = FALSE,
  suffixes = c("_dlib", "_facepp")
)

if (nrow(joined_df) != 59) {
  stop(sprintf("Join per Filename_norm non ha 59 record: %d", nrow(joined_df)))
}

n_common <- nrow(protocol_df)
landmark_rows <- vector("list", n_common)

for (i in seq_len(n_common)) {
  dlib_id <- protocol_df$Dlib_id[i]
  facepp_id <- protocol_df$Facepp_id[i]

  dlib_x_col <- paste0("X", 2 * dlib_id - 1)
  dlib_y_col <- paste0("X", 2 * dlib_id)
  facepp_x_col <- paste0("y", facepp_id) #map x->y is not a bug, but for some mistery reason landmark position in the all_coordinates.raw file is inverted
  facepp_y_col <- paste0("x", facepp_id)

  missing_cols <- setdiff(
    c(dlib_x_col, dlib_y_col, facepp_x_col, facepp_y_col),
    names(joined_df)
  )
  if (length(missing_cols) > 0) {
    stop(sprintf("Colonne mancanti: %s", paste(missing_cols, collapse = ", ")))
  }

  landmark_rows[[i]] <- data.frame(
    Filename = joined_df$Filename_norm,
    `Dlib Landmark` = dlib_id,
    `Face++ Landmark` = facepp_id,
    `Dlib x` = joined_df[[dlib_x_col]],
    `Dlib y` = joined_df[[dlib_y_col]],
    `Face++ x` = joined_df[[facepp_x_col]],
    `Face++ y` = joined_df[[facepp_y_col]],
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

landmarks_df <- do.call(rbind, landmark_rows)

expected_rows <- 59 * n_common
if (nrow(landmarks_df) != expected_rows) {
  stop(sprintf(
    "Dimensione dataframe finale errata: atteso=%d, trovato=%d",
    expected_rows,
    nrow(landmarks_df)
  ))
}

landmarks_mean_df <- aggregate(
  cbind(`Dlib x`, `Dlib y`, `Face++ x`, `Face++ y`) ~ `Dlib Landmark` + `Face++ Landmark`,
  data = landmarks_df,
  FUN = mean
)

if (nrow(landmarks_mean_df) != n_common) {
  stop(sprintf(
    "Dimensione dataframe medie errata: atteso=%d, trovato=%d",
    n_common,
    nrow(landmarks_mean_df)
  ))
}

landmarks_mean_df$`Euclidean distance` <- sqrt(
  (landmarks_mean_df$`Dlib x` - landmarks_mean_df$`Face++ x`)^2 +
  (landmarks_mean_df$`Dlib y` - landmarks_mean_df$`Face++ y`)^2
)
