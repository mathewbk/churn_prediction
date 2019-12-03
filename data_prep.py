import sys
import os
import json
import urllib
import urlparse
import re
import unicodedata
from datetime import datetime

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.streaming import StreamingContext
from pyspark.streaming import *
from pyspark.streaming.dstream import DStream

from pyspark.sql import *
from pyspark.sql.types import *

process_date = sys.argv[1]

conf = (SparkConf()
         .setAppName("bh_fact_insert_checkout"))
'''
         .set("spark.executor.instances", "60")
         .setAppName("ti")
         .set("spark.executor.cores", 4)
         .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
         .set("spark.io.compression.codec", "snappy")
         .set("spark.rdd.compress", "true")
         .set("spark.driver.memory", "40g")
         .set("spark.executor.memory", "50g"))
'''
sc = SparkContext(conf = conf)
sqlContext = HiveContext(sc)

sql = """
use traffic_poc
"""
sqlContext.sql(sql)

sql = """
set hive.exec.dynamic.partition=true
"""
sqlContext.sql(sql)

sql = """
set hive.exec.dynamic.partition.mode=nonstrict
"""
sqlContext.sql(sql)

sql = """
set hive.exec.compress.output=true
"""
sqlContext.sql(sql)

#sql = """
#set hive.execution.engine=spark
#"""
#sqlContext.sql(sql)


sql = """
INSERT OVERWRITE TABLE bh_fact
PARTITION(event, dt, hr)
SELECT
   event_time,
   '' as client_event_time,
   event_id,
   parent_event_id,
   session_id,
   user_browser_id,
   client_ip_address,
   page_id,
   parent_page_id,
   page_url,
   page_path,
   page_hostname,
   page_channel,
   page_country,
   page_division,
   page_type,
   page_app,
   page_domain,
   user_uuid,
   user_agent,
   user_permalink,
   user_logged_in,
   user_device,
   user_device_type,
   (case when deal_permalink_ig <> '' then deal_permalink_ig else deal_permalink end) as deal_permalink,
   '' as deal_option_id,
   deal_uuid,
   order_id,
   order_uuid,
   parent_order_uuid,
   parent_order_id,
   referral_url as referrer_url,
   search_term as referrer_search_term,
   referrer_domain,
   utm_campaign,
   utm_source,
   utm_medium,
   mktg_campaign,
   utm_campaign_channel,
   utm_campaign_strategy,
   utm_campaign_inventory,
   utm_campaign_brand,
   mktg_adgroup,
   mktg_ad_matchtype,
   email_position,
   browser,
   browser_version,
   user_agent_os as os,
   '' as widget_name,
   '' as widget_content_name,
   '' as widget_content_type,
   '' as widget_content_position,
   '' as widget_content_typepos,
   '' as secondary_widgets,
   '' as shopping_cart_uuid,
   '' as cart_contents,
   CASE WHEN ( user_agent LIKE '%bot%'  OR user_agent LIKE '%crawl%'  OR user_agent LIKE '%spider%'  OR user_agent LIKE '%spyder%'  OR user_agent LIKE '%adsbot%'  OR  user_agent    LIKE '%bingbot%'  OR user_agent LIKE '%googlebot%'  OR user_agent LIKE '%slurpr%'  OR user_agent LIKE '%netsparker%'  OR user_agent LIKE '%groupon seo%'  OR user_agent LIKE '%ia_archiver%'  OR user_agent LIKE '%wget%'  OR user_agent LIKE '%curl%'   OR user_agent  LIKE '%groupon-qa-spiderman%' ) THEN 1 ELSE 0 END bot_flag,
   (CASE WHEN user_agent_os IN ('android', 'ios', 'ipad', 'iphone/ipod', 'windows phone') THEN 1
   WHEN user_agent_os LIKE 'blackberry%' THEN 1
   WHEN browser = 'amazon silk' THEN 1
   WHEN browser LIKE '%mobile%' THEN 1
   ELSE 0
   END) mobile_flag,
   internal_ip_ind,
   page_campaign,
   '' as widget_campaign,
   '' as widget_content_campaign,
   user_platform as platform,
   'checkout' as event,
   event_date as dt,
   event_hour as hr
FROM ig_checkout_parsed where  event_date = '2015-10-15' AND event_hour = '00'
""" 
sqlContext.sql(sql)

