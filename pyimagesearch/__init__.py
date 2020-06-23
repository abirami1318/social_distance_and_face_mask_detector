import datetime
k = datetime.datetime.now()
hour = k.hour
if hour >=5 and hour <12:
    status = 'Morning'
elif hour >=12 and hour <4:
    status = 'Afternoon'
elif hour >=4 and hour <8:
    status = 'Evening'
else:
    status = 'Day'
print(' '*50 + 'Good {}!!!'.format(status))

    
