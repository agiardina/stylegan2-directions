# Minimal PCA plotting script converted from analysis/pca.qmd.
# Run from project root:
#   Rscript analysis/pca.R
#
# Inputs:
# - out/latentspaces.csv
# - data/gender.csv
# - out/images/*_projected.jpg
#
# Outputs:
# - out/analysis/pca_sex_photos.png
# - out/analysis/pca_sex_photos.tiff

library(ggplot2)
library(ggrepel)
library(ggimage)
library(dplyr)
library(gridExtra)

ls <- read.csv("out/latentspaces.csv")
lsL <- ls[, 1, drop = FALSE]
lsM <- as.matrix(ls[, -1])

start_layer <- 1
end_layer <- 18
lsM <- lsM[, ((start_layer - 1) * 512) + 1:(end_layer * 512)]

pcout <- prcomp(lsM, scale = FALSE, center = TRUE, rank = 5)
df_pc <- cbind(lsL, pcout$x)

df_gender <- read.csv("data/gender.csv")
df_gender$id <- as.integer(substr(df_gender$Filename, 1, nchar(df_gender$Filename) - 4))
df_gender$Gender[df_gender$Gender == "f"] <- 1
df_gender$Gender[df_gender$Gender == "m"] <- -1
df_gender$Gender <- as.integer(df_gender$Gender)
df_gender <- select(df_gender, id, Gender)

df <- merge(df_pc, df_gender)
df$Sex[df$Gender == 1] <- "F"
df$Sex[df$Gender == -1] <- "M"

output_dir <- "out/analysis"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

options(ggrepel.max.overlaps = Inf)

p_sex <- ggplot(df, aes(x = PC1, y = PC2, col = Sex, label = id)) +
  geom_point() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
  )

ggsave(
  filename = file.path(output_dir, "pca_sex_photos.png"),
  plot = {
    df$image <- paste("out/images/", df$id, "_projected.jpg", sep = "")
    p_faces <- ggplot(df, aes(x = PC1, y = PC2, label = id)) +
      geom_image(aes(image = image), size = .06) +
      theme(
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
      )
    grid.arrange(p_sex + ggtitle("A"), p_faces + ggtitle("B"), ncol = 1)
  },
  width = 2250, height = 2625, dpi = 300, units = "px",
  device = grDevices::png, limitsize = FALSE
)

ggsave(
  filename = file.path(output_dir, "pca_sex_photos.tiff"),
  plot = {
    df$image <- paste("out/images/", df$id, "_projected.jpg", sep = "")
    p_faces <- ggplot(df, aes(x = PC1, y = PC2, label = id)) +
      geom_image(aes(image = image), size = .06) +
      theme(
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
      )
    grid.arrange(p_sex + ggtitle("A"), p_faces + ggtitle("B"), ncol = 1)
  },
  width = 2250, height = 2625, dpi = 300, units = "px",
  device = grDevices::tiff, limitsize = FALSE
)
