
facepp_df <- read.table("../data/all_coordinates.raw", header = TRUE)
dlib_df <- read.csv("../data/landmarks68.csv", sep = ",")

aligned_pattern <- "p_?r_?o_?j_?e_?c_?t_?e_?d"

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
  ifelse(grepl("^[0-9]+$", num), paste0(num, "_projected"), NA_character_)
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

# Per-landmark distance for each image (pixel units)
landmarks_df$`Euclidean distance` <- sqrt(
  (landmarks_df$`Dlib x` - landmarks_df$`Face++ x`)^2 +
  (landmarks_df$`Dlib y` - landmarks_df$`Face++ y`)^2
)

landmarks_mean_df <- aggregate(
  cbind(`Dlib x`, `Dlib y`, `Face++ x`, `Face++ y`, `Euclidean distance`) ~ `Dlib Landmark` + `Face++ Landmark`,
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

# ---- Overlay: exact Dlib landmarks for 297_aligned ----
# NOTE: the Dlib coordinates in landmarks68.csv are pixel coordinates.
# If the points look shifted, use the corresponding *_aligned image.
library(jpeg)
library(readxl)

img_path <- "../img/297.jpg"
img_path <- "/home/agiardina/dev/stylegan2-directions/output/297_aligned.jpg"
img <- readJPEG(img_path)
img_h <- dim(img)[1]
img_w <- dim(img)[2]

dlib_row <- dlib_df[dlib_df$Filename == "297_aligned.jpg", ]
if (nrow(dlib_row) != 1) {
  stop(sprintf("Trovate %d righe per 297_aligned in dlib_df", nrow(dlib_row)))
}

x_cols <- paste0("X", seq(1, 135, 2))
y_cols <- paste0("X", seq(2, 136, 2))

dlib_points <- data.frame(
  `Dlib Landmark` = 1:68,
  `Dlib x` = as.numeric(dlib_row[1, x_cols]),
  `Dlib y` = as.numeric(dlib_row[1, y_cols]),
  check.names = FALSE
)

landmarks_path <- "../data/dlib_facepp_distance.xlsx"
distance_df <- read_excel(landmarks_path, col_names = FALSE)
colnames(distance_df) <- c("Dlib Landmark", "Face++ mean distance")
distance_df <- subset(distance_df, !is.na(`Dlib Landmark`))
distance_df$`Dlib Landmark` <- as.integer(distance_df$`Dlib Landmark`)

landmarks_keep <- distance_df$`Dlib Landmark`
if (length(landmarks_keep) == 0) {
  stop("Nessun landmark trovato in dlib_facepp_distance.xlsx")
}

dlib_points <- merge(
  dlib_points,
  distance_df,
  by = "Dlib Landmark",
  all = FALSE,
  sort = FALSE
)

dir.create("out", showWarnings = FALSE, recursive = TRUE)
out_path <- file.path("out", "297_dlib_landmarks_exact_overlay.png")

png(out_path, width = img_w, height = img_h)
par(mar = c(0, 0, 0, 0))
plot(
  NA,
  xlim = c(0, img_w), ylim = c(img_h, 0),
  asp = 1, xaxs = "i", yaxs = "i",
  xaxt = "n", yaxt = "n", xlab = "", ylab = "", bty = "n"
)
rasterImage(img, 0, img_h, img_w, 0)

symbols(
  dlib_points$`Dlib x`, dlib_points$`Dlib y`,
  circles = dlib_points$`Face++ mean distance`,
  inches = FALSE,
  add = TRUE,
  fg = adjustcolor("deeppink3", alpha.f = 0.35),
  bg = adjustcolor("gray", alpha.f = 0.38),
  lwd = 2
)

points(
  dlib_points$`Dlib x`, dlib_points$`Dlib y`,
  pch = 21, cex = 2.2,
  bg = adjustcolor("red", alpha.f = 0.55),
  col = adjustcolor("white", alpha.f = 0.9),
  lwd = 2
)

# # landmark id labels near each point
# text(
#   dlib_points$`Dlib x`, dlib_points$`Dlib y`,
#   labels = dlib_points$`Dlib Landmark`,
#   pos = 4, offset = 0.35, cex = 1.4,
#   col = adjustcolor("black", alpha.f = 0.85)
# )
# text(
#   dlib_points$`Dlib x`, dlib_points$`Dlib y`,
#   labels = dlib_points$`Dlib Landmark`,
#   pos = 4, offset = 0.35, cex = 1.4,
#   col = adjustcolor("white", alpha.f = 0.95)
# )

dev.off()
message("Wrote: ", out_path)
