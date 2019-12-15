#!/bin/bash

watch "wc data/rolling_avgs.txt && tail data/rolling_avgs.txt && cat data/settings.txt"
