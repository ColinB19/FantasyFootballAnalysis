# this script will be used to pull fantasy football projections from various sites
# check out https://fantasyfootballanalytics.net/2017/03/best-fantasy-football-projections-2017.html
# install.packages("dplyr")    # alternative installation of the %>%
library(dplyr)    # alternatively, this also loads %>%
library(magrittr)
library("ffanalytics")

test_scrape <- scrape_data(src = c("CBS", "ESPN", "Yahoo"), 
                              pos = c("QB", "RB", "WR", "TE", "DST"),
                              season = 2023, week = 6)

# ESPN only works week-to-week back to 2019
ffanalytics:::scrape_espn(pos = c("QB", "RB", "WR", "TE", "DST"), season = 2018, week = 6)
# cbs works
ffanalytics:::scrape_cbs(pos = c("QB", "RB", "WR", "TE", "DST"), season = 2018, week = 6)


test_projections <- projections_table(test_scrape)


View(test_projections)

library(data.table)
that_df <- as.data.frame(rbindlist(test_scrape['QB'], fill = TRUE))

View(that_df)

# this sequence adds risk, ADP, etc
test_projections <- test_projections %>% 
  add_ecr() %>% 
  add_adp() %>% 
  add_aav() %>%
  add_uncertainty()

test_projections <- test_projections %>% 
  add_player_info()

View(test_projections)

?projections_table

View(test_scrape)