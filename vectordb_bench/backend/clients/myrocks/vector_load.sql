ALTER USER 'root'@'localhost' IDENTIFIED BY "150131";
CREATE DATABASE IF NOT EXISTS `test`;
USE `test`;

CREATE TABLE `poi` (
    `coordinate` point NOT NULL SRID 4326,
    `id` int NOT NULL,
    `review_id` varchar(30) NOT NULL,
    `text` text NOT NULL,
    `business_id` varchar(30) NOT NULL,
    `stars` float NOT NULL,
    `date` timestamp NOT NULL,
    `user_id` varchar(30) NOT NULL,
    `city_id` int NOT NULL,
    `text_embedding` json NOT NULL FB_VECTOR_DIMENSION 128,
    PRIMARY KEY (id) COMMENT 'cfname=cf1',
    INDEX key2(text_embedding) FB_VECTOR_INDEX_TYPE 'lsmidx' COMMENT 'cfname=cf1'
) ENGINE=ROCKSDB;

SET GLOBAL local_infile = 1;

LOAD DATA LOCAL INFILE '/home/kevin/experiment/processed/poi_0.csv'
INTO TABLE poi
FIELDS
  TERMINATED BY ','
  OPTIONALLY ENCLOSED BY '"'
  ESCAPED BY '\\'
LINES
  TERMINATED BY '\n'
IGNORE 1 ROWS
(@id,
 @review_id,
 @txt,
 @business_id,
 @stars,
 @dt,
 @user_id,
 @coord,
 @city_id,
 @embedding)
SET
  id             = @id,
  review_id      = @review_id,
  `text`         = @txt,
  business_id    = @business_id,
  stars          = @stars,
  `date`         = STR_TO_DATE(@dt, '%Y-%m-%d %H:%i:%s'),
  user_id        = @user_id,
  coordinate     = ST_GeomFromText(@coord, 4326),
  city_id        = @city_id,
  text_embedding = CAST(@embedding AS JSON);

LOAD DATA LOCAL INFILE '/home/kevin/experiment/processed/poi_1.csv'
INTO TABLE poi
FIELDS
  TERMINATED BY ','
  OPTIONALLY ENCLOSED BY '"'
  ESCAPED BY '\\'
LINES
  TERMINATED BY '\n'
IGNORE 1 ROWS
(@id,
 @review_id,
 @txt,
 @business_id,
 @stars,
 @dt,
 @user_id,
 @coord,
 @city_id,
 @embedding)
SET
  id             = @id,
  review_id      = @review_id,
  `text`         = @txt,
  business_id    = @business_id,
  stars          = @stars,
  `date`         = STR_TO_DATE(@dt, '%Y-%m-%d %H:%i:%s'),
  user_id        = @user_id,
  coordinate     = ST_GeomFromText(@coord, 4326),
  city_id        = @city_id,
  text_embedding = CAST(@embedding AS JSON);

LOAD DATA LOCAL INFILE '/home/kevin/experiment/processed/poi_2.csv'
INTO TABLE poi
FIELDS
  TERMINATED BY ','
  OPTIONALLY ENCLOSED BY '"'
  ESCAPED BY '\\'
LINES
  TERMINATED BY '\n'
IGNORE 1 ROWS
(@id,
 @review_id,
 @txt,
 @business_id,
 @stars,
 @dt,
 @user_id,
 @coord,
 @city_id,
 @embedding)
SET
  id             = @id,
  review_id      = @review_id,
  `text`         = @txt,
  business_id    = @business_id,
  stars          = @stars,
  `date`         = STR_TO_DATE(@dt, '%Y-%m-%d %H:%i:%s'),
  user_id        = @user_id,
  coordinate     = ST_GeomFromText(@coord, 4326),
  city_id        = @city_id,
  text_embedding = CAST(@embedding AS JSON);

LOAD DATA LOCAL INFILE '/home/kevin/experiment/processed/poi_3.csv'
INTO TABLE poi
FIELDS
  TERMINATED BY ','
  OPTIONALLY ENCLOSED BY '"'
  ESCAPED BY '\\'
LINES
  TERMINATED BY '\n'
IGNORE 1 ROWS
(@id,
 @review_id,
 @txt,
 @business_id,
 @stars,
 @dt,
 @user_id,
 @coord,
 @city_id,
 @embedding)
SET
  id             = @id,
  review_id      = @review_id,
  `text`         = @txt,
  business_id    = @business_id,
  stars          = @stars,
  `date`         = STR_TO_DATE(@dt, '%Y-%m-%d %H:%i:%s'),
  user_id        = @user_id,
  coordinate     = ST_GeomFromText(@coord, 4326),
  city_id        = @city_id,
  text_embedding = CAST(@embedding AS JSON);

LOAD DATA LOCAL INFILE '/home/kevin/experiment/processed/poi_4.csv'
INTO TABLE poi
FIELDS
  TERMINATED BY ','
  OPTIONALLY ENCLOSED BY '"'
  ESCAPED BY '\\'
LINES
  TERMINATED BY '\n'
IGNORE 1 ROWS
(@id,
 @review_id,
 @txt,
 @business_id,
 @stars,
 @dt,
 @user_id,
 @coord,
 @city_id,
 @embedding)
SET
  id             = @id,
  review_id      = @review_id,
  `text`         = @txt,
  business_id    = @business_id,
  stars          = @stars,
  `date`         = STR_TO_DATE(@dt, '%Y-%m-%d %H:%i:%s'),
  user_id        = @user_id,
  coordinate     = ST_GeomFromText(@coord, 4326),
  city_id        = @city_id,
  text_embedding = CAST(@embedding AS JSON);

LOAD DATA LOCAL INFILE '/home/kevin/experiment/processed/poi_5.csv'
INTO TABLE poi
FIELDS
  TERMINATED BY ','
  OPTIONALLY ENCLOSED BY '"'
  ESCAPED BY '\\'
LINES
  TERMINATED BY '\n'
IGNORE 1 ROWS
(@id,
 @review_id,
 @txt,
 @business_id,
 @stars,
 @dt,
 @user_id,
 @coord,
 @city_id,
 @embedding)
SET
  id             = @id,
  review_id      = @review_id,
  `text`         = @txt,
  business_id    = @business_id,
  stars          = @stars,
  `date`         = STR_TO_DATE(@dt, '%Y-%m-%d %H:%i:%s'),
  user_id        = @user_id,
  coordinate     = ST_GeomFromText(@coord, 4326),
  city_id        = @city_id,
  text_embedding = CAST(@embedding AS JSON);

LOAD DATA LOCAL INFILE '/home/kevin/experiment/processed/poi_6.csv'
INTO TABLE poi
FIELDS
  TERMINATED BY ','
  OPTIONALLY ENCLOSED BY '"'
  ESCAPED BY '\\'
LINES
  TERMINATED BY '\n'
IGNORE 1 ROWS
(@id,
 @review_id,
 @txt,
 @business_id,
 @stars,
 @dt,
 @user_id,
 @coord,
 @city_id,
 @embedding)
SET
  id             = @id,
  review_id      = @review_id,
  `text`         = @txt,
  business_id    = @business_id,
  stars          = @stars,
  `date`         = STR_TO_DATE(@dt, '%Y-%m-%d %H:%i:%s'),
  user_id        = @user_id,
  coordinate     = ST_GeomFromText(@coord, 4326),
  city_id        = @city_id,
  text_embedding = CAST(@embedding AS JSON);