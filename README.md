# ![Woodbridge (Custom)](https://cdn.nba.com/teams/legacy/www.nba.com/timberwolves/sites/timberwolves/files/towns-signing.jpg) NBA Contacts Analysis

My name is Taneesh Amin, and I am a Freshman Student who is planning on majoring in Computer Science and/or Data Science with minors with Business Administration and Sport Management. 

In this project I go through a thorough process of garthering data through various websites such as basketball reference and crafted NBA. Before I do continue with this project, I do want to acknowledge and thank these two websites for their open source data. I give all the credit to any data I got for this project to them and the NBA API.

## Methedology

This project is based on things I consider valuable. The best ability is availability, so a heavily factored element of this project is the minutes played per game in the playoffs. The next factored feature is offensive production. I measured this first through Points Per game. Then I measured it through usage rate. Regarding defense, I created a STOCKS variable which was the total steals + blocks per game. I believe this is more useful in evaluating both guards and big men. Lastly, I wanted to put some emphasis on analytics. I took advanced offensive analytics such as ORTG and advanced defensive analytics such as ORPTR, and made value charts based off that. 

My regression models were mostly linear and exponential. I often created two different lines of best fits, one to represent those making beow 10 or 20 million and one for those who make above. The reasoning for this is those who make less tend to be rookie contracts which makes the sample heavily biased against stars.

## Examples
![Minutes Played Graph](https://media.discordapp.net/attachments/339767881241722900/1134536805740466176/playoffMP.png?width=1280&height=960)
![Usage Rate Graph](https://media.discordapp.net/attachments/339767881241722900/1134536805446856794/playoffUSG.png?width=1280&height=960)
![PPG Graph](https://media.discordapp.net/attachments/339767881241722900/1134536805199384656/playoffPTS.png?width=1280&height=960)
![STOCKS Graph](https://media.discordapp.net/attachments/339767881241722900/1134536804889022504/playoffSTOCKS.png?width=1280&height=960)
![Off Analytics Graph](https://media.discordapp.net/attachments/339767881241722900/1134536804167589969/offADJ.png?width=1280&height=960)
![Def Analytics Graph](https://media.discordapp.net/attachments/339767881241722900/1134536804461195344/defADJ.png?width=1280&height=960)




## Findings

I used plotly to graph each players expected salary, and I founded the best way to group the players was by team. There are only 16 teams due to the stats being purely playoffs.

[Plotly to Team by Team Analyis](https://taneesh04.github.io/nbaContractProject/)

