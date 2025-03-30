---
layout: post
title: "Political Stances: Association Rule Mining Code"
categories: projects
published: true
in_feed: false
---
 <section>
    <div class="row">
        <div class="col-6 col-12-small">
            <ul class="actions" style="display: flex; gap: 10px; list-style: none; padding: 0;">
                <li><a href="https://nataliermcastro.github.io/projects/2025/01/14/political-stances.html" class="button fit small">Navigate to Project Page</a></li>
                <li><a href="https://nataliermcastro.github.io/projects/2025/03/15/political-stances-arm.html" class="button fit small">Navigate to ARM Page</a></li>
            </ul>
        </div>
    </div> 
</section> 

 <section>
<ul class="actions">
		<li><a href="https://drive.google.com/uc?export-download&id=1F3mMoKVJo9gyNN4bz8GpG8Y4EwGFJEpL" class="buttonprimary icon fa-download">Download R Workspace</a></li>
	</ul>
 </section>

 

 <section>
<ul class="actions">
		<li><a href="https://drive.google.com/uc?export-download&id=1m96NQJZ5jT6T1z5k9g1rF1wsYSFCQF-s" class="buttonprimary icon fa-download">Download R Code</a></li>
	</ul>
 </section>

---
``` r
##########################
 1. ENVIRONMENT CREATION --------------------------
##########################
install.packages("arules", dependencies = TRUE)
install.packages("TSP")
install.packages("data.table")
install.packages("arulesViz", dependencies = TRUE)
install.packages("sp")
install.packages("dplyr", dependencies = TRUE)
install.packages("purrr", dependencies = TRUE)
install.packages("devtools", dependencies = TRUE)
install.packages("tidyr")

install.packages("devtools")
devtools::install_github("mhahsler/arules")

Sys.which("make")

library(Matrix)
library(viridis)
require(arules)
library(TSP)
library(data.table)
library(tcltk)
library(dplyr)
library(devtools)
library(purrr)
library(tidyr)
library(arulesViz)
library(RColorBrewer)

## Setting the working directory
setwd("C:\\Users\\natal\\OneDrive\\university\\info 5653")

##########################
# 2. DATA IMPORT           --------------------------
##########################
basket_data <- read.transactions("Basket Data.csv",
                                 rm.duplicates=FALSE,
                                 format='basket',
                                 header = FALSE,
                                 sep=',',
                                 cols=NULL)

inspect(basket_data[0:1])

##########################
# 3. ASSOCIATION ROLE MINING  --------------------------
##########################

apriori_rules <- arules::apriori(basket_data,
                                parameter = list(support=.05,
                                                 confidence=.15,
                                                 minlen=2,maxlen=5))
inspect(apriori_rules)

## Plotting the most frequent items

arules::itemFrequencyPlot(basket_data,topN=20,
                          col=brewer.pal(8, 'Pastel2'),
                          main = 'Relative Item Frequency Plot',
                          type='relative',
                          ylab="Item Frequency (Relative)")

## Sorting the rules
rules_sorted <- sort(apriori_rules,by='lift',
                     decreasing=TRUE)

inspect(rules_sorted[1:10])

## Saving the rules to a CSV
write(rules_sorted,
      file='Full Association Rules.csv',
      sep= ',',
      quote=TRUE,
      row.names = FALSE)

## Visualizing the rules
plot(rules_sorted[1:200],method='graph',engine='interactive',shading='confidence')

##########################
# 3. ASSOCIATION RULE MINING  --------------------------
#
# Climate Specific Rules 
##########################

## Climate Specific Rules
climate_rules <- arules::apriori(data=basket_data,
                                 parameter = list(support=.001,conf=.05,minlen=2,maxlen=8),
                                 appearance= list(lhs="climate"),
                                 control = list(verbose=TRUE))

inspect(climate_rules)

climate_rules_sorted <- sort(climate_rules,by='lift',
                             decreasing=TRUE)

inspect(climate_rules_sorted[0:10])


## Saving the rules to a CSV
write(climate_rules_sorted,
      file='Climate Association Rules.csv',
      sep= ',',
      quote=TRUE,
      row.names = FALSE)


## Visualizing the rules
plot(climate_rules_sorted,method='graph',shading='confidence')
plot(climate_rules_sorted,method='graph',shading='confidence',engine='interactive')

##########################
# 3. ASSOCIATION RULE MINING  --------------------------
#
# Democrat Specific Rules 
##########################

## Democrat Specific Rules
democrat_rules <- arules::apriori(data=basket_data,
                                 parameter = list(support=.001,conf=.05,minlen=2,maxlen=8),
                                 appearance= list(lhs="democrat"),
                                 control = list(verbose=TRUE))

inspect(democrat_rules)

democrat_rules_sorted <- sort(democrat_rules,by='lift',
                             decreasing=TRUE)

inspect(democrat_rules_sorted[0:10])

## Saving the rules to a CSV
write(democrat_rules_sorted,
      file='Democrat Association Rules.csv',
      sep= ',',
      quote=TRUE,
      row.names = FALSE)

blue_palette <- colorRampPalette(c("lightblue", "blue", "darkblue"))
## Visualizing the rules
plot(democrat_rules_sorted,method='graph',shading='confidence',control = list(colors = blue_palette(10)))


##########################
# 3. ASSOCIATION RULE MINING  --------------------------
#
# Republican Specific Rules 
##########################

## Democrat Specific Rules
republican_rules <- arules::apriori(data=basket_data,
                                  parameter = list(support=.001,conf=.05,minlen=2,maxlen=8),
                                  appearance= list(lhs="republican"),
                                  control = list(verbose=TRUE))

inspect(republican_rules)

republican_rules_sorted <- sort(republican_rules,by='lift',
                              decreasing=TRUE)

inspect(republican_rules_sorted[0:10])

## Saving the rules to a CSV
write(republican_rules_sorted,
      file='Republican Association Rules.csv',
      sep= ',',
      quote=TRUE,
      row.names = FALSE)

red_palette <- colorRampPalette(c("pink", "red", "darkred"))
## Visualizing the rules
plot(republican_rules_sorted,method='graph',shading='confidence',control = list(colors = red_palette(10)))

##########################
# 3. ASSOCIATION RULE MINING  --------------------------
#
# Republican Specific Rules 
##########################

## Democrat Specific Rules
republican_rules <- arules::apriori(data=basket_data,
                                    parameter = list(support=.001,conf=.05,minlen=2,maxlen=8),
                                    appearance= list(lhs="republican"),
                                    control = list(verbose=TRUE))

inspect(republican_rules)

republican_rules_sorted <- sort(republican_rules,by='lift',
                                decreasing=TRUE)

inspect(republican_rules_sorted[0:10])

## Saving the rules to a CSV
write(republican_rules_sorted,
      file='Republican Association Rules.csv',
      sep= ',',
      quote=TRUE,
      row.names = FALSE)

red_palette <- colorRampPalette(c("pink", "red", "darkred"))
## Visualizing the rules
plot(republican_rules_sorted,method='graph',shading='confidence',control = list(colors = red_palette(10)))


##########################
# 3. ASSOCIATION RULE MINING  --------------------------
#
# Republican Specific Rules 
##########################

## Democrat Specific Rules
republican_rules <- arules::apriori(data=basket_data,
                                    parameter = list(support=.001,conf=.05,minlen=2,maxlen=8),
                                    appearance= list(lhs="republican"),
                                    control = list(verbose=TRUE))

inspect(republican_rules)

republican_rules_sorted <- sort(republican_rules,by='lift',
                                decreasing=TRUE)

inspect(republican_rules_sorted[0:10])

## Saving the rules to a CSV
write(republican_rules_sorted,
      file='Republican Association Rules.csv',
      sep= ',',
      quote=TRUE,
      row.names = FALSE)

red_palette <- colorRampPalette(c("pink", "red", "darkred"))
## Visualizing the rules
plot(republican_rules_sorted,method='graph',shading='confidence',control = list(colors = red_palette(10)))


##########################
# 3. ASSOCIATION RULE MINING  --------------------------
#
# Government Specific Rules 
##########################

## Democrat Specific Rules
government_rules <- arules::apriori(data=basket_data,
                                    parameter = list(support=.001,conf=.05,minlen=2,maxlen=8),
                                    appearance= list(lhs="government"),
                                    control = list(verbose=TRUE))

inspect(government_rules)

government_rules_sorted <- sort(government_rules,by='lift',
                                decreasing=TRUE)

inspect(government_rules_sorted[0:10])

## Saving the rules to a CSV
write(republican_rules_sorted,
      file='Government Association Rules.csv',
      sep= ',',
      quote=TRUE,
      row.names = FALSE)

earth_tone_palette <- colorRampPalette(c("beige","brown", "darkolivegreen"))
## Visualizing the rules
plot(government_rules_sorted,method='graph',shading='confidence',control = list(colors = earth_tone_palette(10)))

##########################
# 3. ASSOCIATION RULE MINING  --------------------------
#
# Government Specific Rules 
##########################

## Democrat Specific Rules
government_rules <- arules::apriori(data=basket_data,
                                    parameter = list(support=.001,conf=.05,minlen=2,maxlen=8),
                                    appearance= list(lhs="government"),
                                    control = list(verbose=TRUE))

inspect(government_rules)

government_rules_sorted <- sort(government_rules,by='lift',
                                decreasing=TRUE)

inspect(government_rules_sorted[0:10])

## Saving the rules to a CSV
write(republican_rules_sorted,
      file='Government Association Rules.csv',
      sep= ',',
      quote=TRUE,
      row.names = FALSE)

earth_tone_palette <- colorRampPalette(c("beige","brown", "darkolivegreen"))
## Visualizing the rules
plot(government_rules_sorted,method='graph',shading='confidence',control = list(colors = earth_tone_palette(10)))

##########################
# 3. ASSOCIATION RULE MINING  --------------------------
#
# Science Specific Rules 
##########################

## Democrat Specific Rules
science_rules <- arules::apriori(data=basket_data,
                                 parameter = list(support=.001,conf=.5,minlen=2,maxlen=8),
                                 appearance= list(lhs="scienc"),
                                 control = list(verbose=TRUE))

inspect(science_rules)

science_rules_sorted <- sort(science_rules,by='lift',
                             decreasing=TRUE)

inspect(science_rules_sorted[0:7])

## Saving the rules to a CSV
write(science_rules_sorted,
      file='Science Association Rules.csv',
      sep= ',',
      quote=TRUE,
      row.names = FALSE)

earth_tone_palette <- colorRampPalette(c("beige","brown", "darkolivegreen"))
## Visualizing the rules
plot(science_rules_sorted,method='graph',shading='confidence',control = list(colors = earth_tone_palette(10)))

##########################
# 3. ASSOCIATION RULE MINING  --------------------------
#
# Indigenous Specific Rules 
##########################

## Democrat Specific Rules
indg_rules <- arules::apriori(data=basket_data,
                              parameter = list(support=.001,conf=.05,minlen=2,maxlen=8),
                              appearance= list(lhs="tribal"),
                              control = list(verbose=TRUE))

inspect(science_rules)

indg_rules_sorted <- sort(indg_rules,by='lift',
                          decreasing=TRUE)

inspect(indg_rules_sorted[0:10])

## Saving the rules to a CSV
write(indg_rules_sorted,
      file='Indigenous Association Rules.csv',
      sep= ',',
      quote=TRUE,
      row.names = FALSE)

earth_tone_palette <- colorRampPalette(c("beige","brown", "darkolivegreen"))
## Visualizing the rules
plot(indg_rules_sorted,method='graph',shading='confidence',control = list(colors = earth_tone_palette(10)))
```
