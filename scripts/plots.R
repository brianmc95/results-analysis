#########################################################################
# Brian McCarthy
# b.mccarthy@cs.ucc.ie
# R script designed to plot different graphs and such.
########################################################################

source("scripts/data_preparation.R")

options("scipen"=100, "digits"=4)

# In my eyes the total number of packets is receivedBSMs/Broadcasts + TotalLostPackets

# PDR per node across all runs.

pdr_node <- scalars_df %>%
  group_by(node) %>%
  summarize_at(vars(busyTime:totalTime), mean) %>%
  ggplot() +
  geom_point(mapping = aes(x = node, y = ReceivedBroadcasts/(ReceivedBroadcasts+TotalLostPackets))) +
  
  labs(x="Node", y="PDR",
       title=paste("PDR per node averaged across", summarize_at(scalars_df, vars(repetition), max) + 1, "repetitions", sep=" "))

# PDR per node per run.
pdr_run_node <- scalars_df %>%
  ggplot() +
  geom_point(mapping = aes(x = node, y = ReceivedBroadcasts/(ReceivedBroadcasts+TotalLostPackets),
                           colour = repetition)) +
  
  labs(x="Node", y="PDR",
       title=paste("PDR per node across", summarize_at(scalars_df, vars(repetition), max) + 1, "repetitions", sep=" "))

# PDR at network level.
pdr_net_run <- scalars_df %>%
  group_by(repetition) %>%
  summarize_at(vars(busyTime:totalTime), mean) %>%
  ggplot() +
  geom_bar(mapping = aes(x = repetition, y = ReceivedBroadcasts/(ReceivedBroadcasts+TotalLostPackets)),
           stat = "identity") +
  labs(x="Repetition", y="PDR",
       title=paste("PDR per network across", summarize_at(scalars_df, vars(repetition), max) + 1, "repetitions", sep=" "))

