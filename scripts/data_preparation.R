#######################################################################################################
# Brian McCarhy
# b.mccarthy@cs.ucc.ie
# This is designed to read OMNeT++ CSV files, split them into their constituent parts
# Such as scalars, vectors, parameters and such and then tidy them so as to be prepared for graphing in 
# subsequent stages
#######################################################################################################

library(tidyverse)

omnetpp_df <- read_csv("data/raw_data/WithBeaconing_e2e_delay.csv") %>%
  
  # Split the run line into it's constituent parts
  separate(run, c("config_name", "repetition", "date", "time", "process_id"),sep="-") %>%
    
  # Split module into it's constituent parts.
  separate(module, c("network", "node", "layer"), sep="\\.", extra="merge", fill="right") %>%
  
  unite(date, time, col="datetime", sep="-") %>%
  
  # Remove the node[] from the node number, just makes things more readable
  mutate_at("node", ~gsub("node", "", .)) %>%
  mutate_at("node", ~gsub("\\[", "", .)) %>%
  mutate_at("node", ~gsub("\\]", "", .)) %>%
  
  # Remove layer as it isn't necessary I beleieve and complicates the data
  select(-c(process_id, datetime, layer)) %>%
  
  # Spread scalar results values
  spread(name, value) %>%
  
  # Convert columns to integers
  mutate(node = as.numeric(node)) %>%
  mutate(repetition = as.numeric(repetition))

types = split(omnetpp_df, omnetpp_df$type) 
    
params_df <- types$param %>% 
  select_if(~!all(is.na(.))) %>%
  select(-one_of("type"))

vectors_df <- types$vector    %>%
  select_if(~!all(is.na(.))) %>%
  # Convert columns to integers
  mutate(node = as.numeric(node)) %>%
  select(-one_of("type"))

histograms_df <- types$histogram %>% 
  select_if(~!all(is.na(.))) %>%
  # Convert columns to integers
  mutate(node = as.numeric(node)) %>%
  select(-one_of("type"))

# Grab only the scalars
scalars_df <- types$scalar %>%
  # Remove the manager node as it does not have interesting information associated with it.
  filter(node != "manager") %>%
  # Remove empty columns (not corresponding to this variable type)
  select_if(~!all(is.na(.))) %>%
  
  select(-one_of("type"))
  
    

rm(omnetpp_df, types)

# rm (params, runnattr, attr, vectors, histograms, scalars)
