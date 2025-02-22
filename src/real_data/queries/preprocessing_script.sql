
------------------------------------------------------------------------
------------------Coping data to a new table----------------------------
------------------------------------------------------------------------
--- Create table for initially required data

-- CREATE TABLE "data_needed_uncleaned" (
-- 	"secid"				REAL, --- INTEGER
-- 	"symbol"			TEXT,
-- 	"date"				DATE,
-- 	"exdate"			DATE,
-- 	"last_date"			DATE,
-- 	"best_bid"			REAL,
-- 	"best_offer"		REAL,
-- 	"volume"			REAL,
-- 	"open_interest"		REAL,
-- 	"cp_flag"			TEXT,
-- 	"impl_volatility"	REAL,
-- 	"strike_price"		REAL
-- );

-- -- Insert required values
-- INSERT INTO data_needed_uncleaned(secid, symbol, date, exdate, cp_flag, best_bid, best_offer, strike_price, impl_volatility, volume, open_interest)
-- SELECT secid, symbol, date, exdate, cp_flag, best_bid, best_offer, strike_price, impl_volatility, volume, open_interest 
-- FROM option_price_table;

----------------------------------------------------------------------------
------------------Adding new columns to option_price_table------------------
----------------------------------------------------------------------------

-- Insert spot prices into overarching table
ALTER TABLE option_price_table
ADD COLUMN spot_price REAL;

-- UPDATE option_price_table
-- SET spot_price = (
--     SELECT close
--     FROM stock_price_table AS stock_prices
--     WHERE stock_prices.secid = option_price_table.secid
--     AND stock_prices.date = option_price_table.date
-- )
-- WHERE EXISTS (
--     SELECT 1
--     FROM stock_price_table AS stock_prices
--     WHERE stock_prices.secid = option_price_table.secid
--     AND stock_prices.date = option_price_table.date
-- );

UPDATE option_price_table
SET spot_price = stock_prices.close
FROM stock_price_table AS stock_prices
WHERE stock_prices.secid = option_price_table.secid
AND stock_prices.date = option_price_table.date;

-- CORRECT strike_price values (initial values are K*1000)
UPDATE option_price_table
SET strike_price = strike_price / 1000;

-- Option price
ALTER TABLE option_price_table
ADD COLUMN option_price REAL;

UPDATE option_price_table
SET option_price = (best_bid + best_offer) / 2;

-- Calculate value of option price in units of the strike price
ALTER TABLE option_price_table
ADD COLUMN V REAL;

UPDATE option_price_table
SET V = option_price / strike_price;

-- Moneyness
ALTER TABLE option_price_table
ADD COLUMN moneyness REAL;

UPDATE option_price_table
SET moneyness = spot_price / strike_price;

-- REMOVE data deep in or out of the money
DELETE FROM option_price_table
WHERE moneyness < 0.5 OR moneyness > 1.5;

-- Add tau and days columns and calculate time to expiry
ALTER TABLE option_price_table
ADD COLUMN tau REAL,
ADD COLUMN days REAL;

---- Calculate days to maturity, and tau (time to maturity in years)
UPDATE option_price_table
SET days = julianday(exdate) - julianday(date);

UPDATE option_price_table
SET tau = days / 365;

-- id as row number
ALTER TABLE option_price_table
ADD COLUMN id INT;

UPDATE option_price_table
SET id = ROWID;

-- Get interest rates
ALTER TABLE option_price_table
ADD COLUMN rf REAL;

-- Where IR data points are available without interpolation
UPDATE option_price_table AS opt
SET rf = ir.rate
FROM zerocd AS ir
WHERE opt.date = ir.date
AND opt.days = ir.days
AND ir.rate IS NOT NULL;

-- Create joint indices for faster computation
CREATE INDEX idx_options_date_tau ON option_price_table(date, days);
CREATE INDEX idx_interest_rates_date_days ON zerocd(date, days);

-- Create a table for the data points with interpolated interest rates (where option maturity is between
-- two interest rate maturities.
CREATE TABLE interpolated_data AS
SELECT
	opt.id,
	opt.date,
	opt.days AS t,
	r1.rate AS ir1,
	r1.days AS t1,
	r2.rate AS ir2,
	r2.days AS t2,
	r1.rate + (r2.rate - r1.rate)/(r2.days - r1.days) * (opt.days - r1.days) AS rate
FROM
	option_price_table AS opt
LEFT JOIN
	zerocd AS r1 ON opt.date = r1.date
	AND r1.days = (
		SELECT MAX(z1.days) FROM zerocd AS z1 WHERE opt.date = z1.date AND z1.days < opt.days
		)
LEFT JOIN
	zerocd AS r2 ON opt.date = r2.date
	AND r2.days = (
		SELECT MIN(z2.days) FROM zerocd AS z2 WHERE opt.date = z2.date AND z2.days > opt.days
		)
WHERE r1.days IS NOT NULL AND r2.days IS NOT NULL;


-- ADD interpolated rates, where option_price_table.rf is NULL
UPDATE option_price_table
SET rf = interpolated_data.rate
FROM interpolated_data
WHERE option_price_table.id = interpolated_data.id
AND option_price_table.rf IS NULL;

------------------------------------------------------------------------
----------------------------- CLEANING ---------------------------------
------------------------------------------------------------------------
-- Create a table data_needed that is initially a copy of data_needed_uncleaned

-- CREATE TABLE clean_table AS
-- SELECT * FROM option_price_table;

-- Create the FINAL table and insert the required values
CREATE TABLE clean_table (
	"secid"		REAL,
	"date"		DATE,
	"S"			REAL,
	"sigma"		REAL,
	"tau"		REAL,
	"r"			REAL,
	"cp_flag"	TEXT,
	"V"			REAL
);

-- Insert values from the cleaned data
INSERT INTO clean_table(secid, date, S, sigma, tau, r, cp_flag, V)
SELECT
	secid,
	date,
	moneyness AS S,
	impl_volatility AS sigma,
	tau,
	rf AS r,
	cp_flag,
	V
FROM option_price_table;

-- NULL values removed
DELETE FROM clean_table
WHERE secid IS NULL
	OR date IS NULL
	OR S IS NULL
	OR sigma IS NULL
	OR tau IS NULL
	OR r IS NULL
	OR cp_flag IS NULL
	OR V IS NULL;

-- -- Create an additional Moneyness column
-- ALTER TABLE final_replication_data
-- ADD COLUMN M REAL;

-- -- Insert the appropriate moneyness values
-- UPDATE final_replication_data
-- SET M = S / K;