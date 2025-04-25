library(dplyr)
library(tidyr)
library(tidyverse)
# our sentences

df = read_csv('./Data/all_sentences.csv')

# Create a column that randomly assigns one of the sentences to group 1 or 2
df = df %>%
  group_by(Item) %>%
  mutate(group = sample(c(1, 2)))

condition1 = df %>%
  filter(group == 1)

condition2 = df %>%
  filter(group == 2)

clipr::write_clip(condition1)
clipr::write_clip(condition2)

condition1a = read_csv('./Data/condition1a.csv') %>%
  mutate(Word = case_when(Asked_about_first_or_second == 1 ~ Word1,
                          Asked_about_first_or_second == 2 ~ Word2))
condition1b = read_csv('./Data/condition1b.csv') %>%
  mutate(Word = case_when(Asked_about_first_or_second == 1 ~ Word1,
                          Asked_about_first_or_second == 2 ~ Word2))
condition2a = read_csv('./Data/condition2a.csv') %>%
  mutate(Word = case_when(Asked_about_first_or_second == 1 ~ Word1,
                          Asked_about_first_or_second == 2 ~ Word2))
condition2b = read_csv('./Data/condition2b.csv') %>%
  mutate(Word = case_when(Asked_about_first_or_second == 1 ~ Word1,
                          Asked_about_first_or_second == 2 ~ Word2))

clipr::write_clip(condition1a)
clipr::write_clip(condition1b)
clipr::write_clip(condition2a)
clipr::write_clip(condition2b)
