
facepp_df <- read.table("../data/all_coordinates.raw", header = TRUE)
dlib_df <- read.csv("../data/landmarks68.csv", sep = ",")

x_cols <- paste0("X", seq(1, 135, 2))
y_cols <- paste0("X", seq(2, 136, 2))

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
img_path <- "/home/agiardina/dev/stylegan2-directions/output/297_projected.jpg"
img <- readJPEG(img_path)
img_h <- dim(img)[1]
img_w <- dim(img)[2]

dlib_row <- dlib_df[dlib_df$Filename == "297_projected.jpg", ]
if (nrow(dlib_row) != 1) {
  stop(sprintf("Trovate %d righe per 297_projected in dlib_df", nrow(dlib_row)))
}

dlib_points <- data.frame(
  `Dlib Landmark` = 1:68,
  `Dlib x` = as.numeric(dlib_row[1, x_cols]),
  `Dlib y` = as.numeric(dlib_row[1, y_cols]),
  check.names = FALSE
)

facepp_row <- facepp_aligned[facepp_aligned$Filename_norm == "297_projected", ]
if (nrow(facepp_row) != 1) {
  stop(sprintf("Trovate %d righe per 297_projected in facepp_aligned", nrow(facepp_row)))
}

protocol_dlib_ids <- unique(protocol_df$Dlib_id)
protocol_facepp_ids <- unique(protocol_df$Facepp_id)

facepp_x_cols <- paste0("y", protocol_facepp_ids)
facepp_y_cols <- paste0("x", protocol_facepp_ids)
missing_facepp_cols <- setdiff(c(facepp_x_cols, facepp_y_cols), names(facepp_row))
if (length(missing_facepp_cols) > 0) {
  stop(sprintf("Colonne Face++ mancanti: %s", paste(missing_facepp_cols, collapse = ", ")))
}

facepp_points <- data.frame(
  `Face++ Landmark` = protocol_facepp_ids,
  `Face++ x` = as.numeric(facepp_row[1, facepp_x_cols]),
  `Face++ y` = as.numeric(facepp_row[1, facepp_y_cols]),
  check.names = FALSE
)

dlib_points <- subset(dlib_points, `Dlib Landmark` %in% protocol_dlib_ids)

landmarks_path <- "../data/dlib_facepp_distance.xlsx"
distance_df <- read_excel(landmarks_path, col_names = FALSE)
colnames(distance_df) <- c("Dlib Landmark", "Face++ mean distance")
distance_df <- subset(distance_df, !is.na(`Dlib Landmark`))
distance_df$`Dlib Landmark` <- as.integer(distance_df$`Dlib Landmark`)

parse_variant_info <- function(filename) {
  projected_match <- regexec("^([0-9]+)_projected\\.jpg$", filename)
  projected_parts <- regmatches(filename, projected_match)[[1]]
  if (length(projected_parts) == 2) {
    return(list(
      Id = projected_parts[2],
      Variant = NA_character_,
      Direction = NA_character_,
      Magnitude = NA_character_,
      Is_projected = TRUE,
      Is_aligned = FALSE
    ))
  }

  aligned_match <- regexec("^([0-9]+)_aligned\\.jpg$", filename)
  aligned_parts <- regmatches(filename, aligned_match)[[1]]
  if (length(aligned_parts) == 2) {
    return(list(
      Id = aligned_parts[2],
      Variant = NA_character_,
      Direction = NA_character_,
      Magnitude = NA_character_,
      Is_projected = FALSE,
      Is_aligned = TRUE
    ))
  }

  variant_match <- regexec("^([0-9]+)_([^_]+)_(neg|pos)([0-9]+)\\.jpg$", filename)
  variant_parts <- regmatches(filename, variant_match)[[1]]
  if (length(variant_parts) == 5) {
    return(list(
      Id = variant_parts[2],
      Variant = variant_parts[3],
      Direction = variant_parts[4],
      Magnitude = variant_parts[5],
      Is_projected = FALSE,
      Is_aligned = FALSE
    ))
  }

  list(
    Id = NA_character_,
    Variant = NA_character_,
    Direction = NA_character_,
    Magnitude = NA_character_,
    Is_projected = FALSE,
    Is_aligned = FALSE
  )
}

variant_info <- lapply(dlib_df$Filename, parse_variant_info)
variant_df <- do.call(rbind, lapply(variant_info, as.data.frame))
dlib_df <- cbind(dlib_df, variant_df, stringsAsFactors = FALSE)

projected_df <- subset(dlib_df, Is_projected)
aligned_df <- subset(dlib_df, Is_aligned)
variants_df <- subset(dlib_df, !Is_projected & !Is_aligned & !is.na(Variant))

if (nrow(projected_df) == 0) {
  stop("Nessuna immagine projected trovata in dlib_df")
}
if (nrow(aligned_df) == 0) {
  stop("Nessuna immagine aligned trovata in dlib_df")
}
if (nrow(variants_df) == 0) {
  stop("Nessuna variante trovata in dlib_df")
}

expand_landmarks <- function(df, id_cols) {
  x_vals <- as.matrix(df[, x_cols])
  y_vals <- as.matrix(df[, y_cols])
  n <- nrow(df)
  landmark_ids <- 1:68
  data.frame(
    df[id_cols][rep(seq_len(n), each = 68), , drop = FALSE],
    `Dlib Landmark` = rep(landmark_ids, times = n),
    `Dlib x` = as.numeric(as.vector(t(x_vals))),
    `Dlib y` = as.numeric(as.vector(t(y_vals))),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

projected_long <- expand_landmarks(projected_df, c("Id"))
aligned_long <- expand_landmarks(aligned_df, c("Id"))
variants_long <- expand_landmarks(variants_df, c("Id", "Variant", "Direction", "Magnitude"))

variant_joined <- merge(
  variants_long,
  projected_long,
  by = c("Id", "Dlib Landmark"),
  suffixes = c("_var", "_proj"),
  all = FALSE,
  sort = FALSE
)

variant_joined$`Euclidean displacement` <- sqrt(
  (variant_joined$`Dlib x_var` - variant_joined$`Dlib x_proj`)^2 +
    (variant_joined$`Dlib y_var` - variant_joined$`Dlib y_proj`)^2
)

aligned_joined <- merge(
  aligned_long,
  projected_long,
  by = c("Id", "Dlib Landmark"),
  suffixes = c("_aligned", "_proj"),
  all = FALSE,
  sort = FALSE
)

aligned_joined$`Euclidean displacement` <- sqrt(
  (aligned_joined$`Dlib x_aligned` - aligned_joined$`Dlib x_proj`)^2 +
    (aligned_joined$`Dlib y_aligned` - aligned_joined$`Dlib y_proj`)^2
)

variant_mean_df <- aggregate(
  `Euclidean displacement` ~ Id + Variant + `Dlib Landmark`,
  data = variant_joined,
  FUN = mean
)

variant_max_df <- aggregate(
  `Euclidean displacement` ~ Id + Variant + `Dlib Landmark`,
  data = variant_joined,
  FUN = max
)

aligned_max_df <- aggregate(
  `Euclidean displacement` ~ Id + `Dlib Landmark`,
  data = aligned_joined,
  FUN = max
)

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

target_id <- "297"
if (!target_id %in% projected_df$Id) {
  target_id <- projected_df$Id[1]
}
if (!target_id %in% variant_mean_df$Id) {
  target_id <- variant_mean_df$Id[1]
}
if (!target_id %in% aligned_max_df$Id) {
  target_id <- aligned_max_df$Id[1]
}

projected_row <- projected_df[projected_df$Id == target_id, ]
if (nrow(projected_row) != 1) {
  stop(sprintf("Trovate %d righe projected per id %s", nrow(projected_row), target_id))
}

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
  fg = adjustcolor("gold", alpha.f = 0.95),
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

points(
  facepp_points$`Face++ x`, facepp_points$`Face++ y`,
  pch = 24, cex = 2.0,
  bg = adjustcolor("dodgerblue3", alpha.f = 0.6),
  col = adjustcolor("white", alpha.f = 0.9),
  lwd = 2
)

legend(
  "bottomleft",
  legend = c("Dlib landmarks", "Face++ landmarks", "Face++ mean distance"),
  pt.cex = c(2.2, 2.0, NA),
  pch = c(21, 24, NA),
  pt.bg = c(
    adjustcolor("red", alpha.f = 0.55),
    adjustcolor("dodgerblue3", alpha.f = 0.6),
    NA
  ),
  col = c(
    adjustcolor("white", alpha.f = 0.9),
    adjustcolor("white", alpha.f = 0.9),
    adjustcolor("gold", alpha.f = 0.95)
  ),
  lwd = c(2, 2, 3),
  lty = c(NA, NA, 1),
  bty = "o",
  bg = adjustcolor("black", alpha.f = 0.6),
  text.col = adjustcolor("white", alpha.f = 0.98),
  cex = 1.3
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

# ---- Overlay: mean displacement per variant vs projected ----
img_path <- file.path("/home/agiardina/dev/stylegan2-directions/output", paste0(target_id, "_projected.jpg"))
if (!file.exists(img_path)) {
  stop(sprintf("Immagine projected non trovata: %s", img_path))
}
img <- readJPEG(img_path)
img_h <- dim(img)[1]
img_w <- dim(img)[2]

projected_points <- data.frame(
  `Dlib Landmark` = 1:68,
  `Dlib x` = as.numeric(projected_row[1, x_cols]),
  `Dlib y` = as.numeric(projected_row[1, y_cols]),
  check.names = FALSE
)

aligned_points <- subset(aligned_max_df, Id == target_id)
aligned_points <- merge(
  projected_points,
  aligned_points,
  by = "Dlib Landmark",
  all = FALSE,
  sort = FALSE
)
aligned_points <- subset(aligned_points, `Dlib Landmark` %in% landmarks_keep)

out_path <- file.path(
  "out",
  sprintf("%s_aligned_projected_max_displacement_overlay.png", target_id)
)

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
  aligned_points$`Dlib x`, aligned_points$`Dlib y`,
  circles = aligned_points$`Euclidean displacement`,
  inches = FALSE,
  add = TRUE,
  fg = adjustcolor("deeppink3", alpha.f = 0.35),
  bg = adjustcolor("blue", alpha.f = 0.38),
  lwd = 2
)

points(
  aligned_points$`Dlib x`, aligned_points$`Dlib y`,
  pch = 21, cex = 2.2,
  bg = adjustcolor("red", alpha.f = 0.55),
  col = adjustcolor("white", alpha.f = 0.9),
  lwd = 2
)

dev.off()
message("Wrote: ", out_path)

variant_levels <- sort(unique(variant_mean_df$Variant[variant_mean_df$Id == target_id]))
for (variant_name in variant_levels) {
  variant_points <- subset(
    variant_mean_df,
    Variant == variant_name & Id == target_id
  )
  variant_points <- merge(
    projected_points,
    variant_points,
    by = "Dlib Landmark",
    all = FALSE,
    sort = FALSE
  )

  variant_points <- subset(variant_points, `Dlib Landmark` %in% landmarks_keep)

  out_path <- file.path(
    "out",
    sprintf("%s_%s_mean_displacement_overlay.png", target_id, variant_name)
  )

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
    variant_points$`Dlib x`, variant_points$`Dlib y`,
    circles = variant_points$`Euclidean displacement`,
    inches = FALSE,
    add = TRUE,
    fg = adjustcolor("deeppink3", alpha.f = 0.35),
    bg = adjustcolor("blue", alpha.f = 0.38),
    lwd = 2
  )

  points(
    variant_points$`Dlib x`, variant_points$`Dlib y`,
    pch = 21, cex = 2.2,
    bg = adjustcolor("red", alpha.f = 0.55),
    col = adjustcolor("white", alpha.f = 0.9),
    lwd = 2
  )

  dev.off()
  message("Wrote: ", out_path)

  variant_points <- subset(
    variant_max_df,
    Variant == variant_name & Id == target_id
  )
  variant_points <- merge(
    projected_points,
    variant_points,
    by = "Dlib Landmark",
    all = FALSE,
    sort = FALSE
  )

  variant_points <- subset(variant_points, `Dlib Landmark` %in% landmarks_keep)

  out_path <- file.path(
    "out",
    sprintf("%s_%s_max_displacement_overlay.png", target_id, variant_name)
  )

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
    variant_points$`Dlib x`, variant_points$`Dlib y`,
    circles = variant_points$`Euclidean displacement`,
    inches = FALSE,
    add = TRUE,
    fg = adjustcolor("deeppink3", alpha.f = 0.35),
    bg = adjustcolor("blue", alpha.f = 0.38),
    lwd = 2
  )

  points(
    variant_points$`Dlib x`, variant_points$`Dlib y`,
    pch = 21, cex = 2.2,
    bg = adjustcolor("red", alpha.f = 0.55),
    col = adjustcolor("white", alpha.f = 0.9),
    lwd = 2
  )

  dev.off()
  message("Wrote: ", out_path)
}
