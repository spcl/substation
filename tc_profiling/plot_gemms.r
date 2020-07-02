library(ggplot2)

#measurements = c("dWlin1", "dWlin2", "dWout", "dWQ", "dWQK-fused", "dWQKV-fused", 
#                 "dX1gamma", "dx1QKT", "dX2gamma", "dX2QKT", "dXlin1", "dXlin2",
#                 "dXout", "dXQ", "dXQK-fused", "dXQKV-fused", "gamma", "KV-fused",
#                 "lin1", "lin2", "out", "Q", "QKT", "QKV-fused")

df = data.frame()

for (f in list.files(pattern="result.csv")) {
  type = unlist(strsplit(f, '-'))[1]
  df_temp = read.csv(file=f,stringsAsFactors = FALSE)
  df_temp$Type = type
  df = rbind(df, df_temp)
}

#df = read.csv(file="dWlin1-result.csv", stringsAsFactors = FALSE)

df$TensorCores = grepl("tc", df$Implementation)
df$Float32 = grepl("32", df$Implementation)

df16 = df[df$Float32 == FALSE, ]
df32 = df[df$Float32 == TRUE, ]

df16 = transform(df16, MM=ifelse(M>N, M, N))
df16 = transform(df16, NN=ifelse(M>N, N, M))
df16 = transform(df16, MNKB=sprintf("M: %d, N: %d\nK: %d, B: %d", MM, NN, K, batch))

df16 = within(df16, { Types <- ave(Type, MNKB, FUN=function(x) toString(unique(x)))})

df16$FLOPs <- 2
df16$FLOPs <- df16$FLOPs * df16$M
df16$FLOPs <- df16$FLOPs * df16$N
df16$FLOPs <- df16$FLOPs * df16$K
df16$FLOPs <- df16$FLOPs * df16$batch

# time in milliseconds
df16 = transform(df16, usedFLOPs=ifelse(TensorCores, 125e12 * Time * 1e-3, 31.4e12 * Time * 1e-3))
df16$ratioOfPeak = df16$FLOPs / df16$usedFLOPs 
df16$percentOfPeak = df16$ratioOfPeak * 100

#x= df16[which.max(df16$percentOfPeak),]

#y=df[387293,]

#unique(df16$Type) 

df16$TensorCoresText = ifelse(df16$TensorCores, "Enabled", "Disabled")



plot = ggplot(df16, aes(x=factor(0), y=percentOfPeak, fill=TensorCoresText))
plot = plot + theme(axis.title.y=element_blank(),
                    axis.text.y=element_blank(),
                    axis.ticks.y=element_blank(),
                    legend.position="bottom")
plot = plot + ylab("% of peak performance") + labs(fill = "Tensor cores (bottom to top)")
plot = plot + geom_violin()
plot = plot + facet_wrap(vars(sprintf("%s\nM: %d, N: %d, K: %d, B: %d", Types, MM, NN, K, batch)), nrow=3)
#plot = plot + facet_wrap(vars(Type), nrow=3, scales="free_y")
#plot = plot + facet_wrap(vars(sprintf("M: %d\nN: %d\n K: %d\n B: %d", M, N, K, batches)), nrow=3)
plot = plot + stat_summary(aes(y=Time, label=sprintf("best: %.2f ms", round(..y..,2))), 
                           fun=min,geom="text", size=3, position = position_dodge(1.1), hjust = -2.2, vjust=-2.0)
plot = plot + stat_summary(aes(y=Time, label=sprintf("worst: %.2f ms", round(..y..,2))), 
                           fun=max,geom="text", size=3, position = position_dodge(1.1), hjust = -0.1, vjust=-2.0)
plot = plot + scale_y_continuous(limits = c(0, 100))
plot = plot + coord_flip()
plot

ggsave("bmm_violins.pdf", plot, width = 12, height = 6)

