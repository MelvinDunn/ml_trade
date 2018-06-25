its trivial to leave that in as a datetitme object
symbols are our 3 or 4 letter symbols
order can be a buy or sell

its gonna be your job simulating the market to execute that order. there will be other orders on other dates.

lets put in another order.

symbol ibm, sell 100

let me see what it looks like.

dokay

its the magic of shorting, if you have a sell order
and you don't own it, that's equivalent to shorting.

we'll talk about the underlying math for this
project in a second.

Are they market orders or limit orders?

What kind of order? sorry I'm writing it small here.

The answer is that they are market on close orders
which means that you get the adjusted close
price for that day. So uh all the stuff that
we're gonna do for this assignment are assuming
adjusted close prices. If you look at the data
you have, you'll see that there are
closing prices and adjusted closing prices.

The difference between those two,
for now the important thing to remember is to
use adjusted close. Before that date the market
actually closed at. Calculate the increase
or decrease in a stock before we held it.

Always assume market on close prices.

One more order - sell the shares of apple
that we had bought earlier.

Example of your input, 

output the history of the value of your input
at a specific point in time.

Information that you are missing -
another piece of data that is coming in
is initial cash.

What'll happen over time, the initial
value of your portfolio is your initial cash

track the value of the portfolio over time.
plus or minus the initial cash value.

you need to provide a history for every trading day
between the first day and the last day of training
inclusive, the way you can discover how big this
history needs to be, you need to find the first day
and the last day, and fill out a dataframe
that fills out all those days.


Step 1.

Understand this basic idea.

Read in the dataframe prices.

We've provided you a method - if you give it the symbols
it will give you a start date and end date, it will
read in this data for you.

You can read this orders file into a dataframe as well
symbols in this file, earliest date, last date.

Includes every single training day between this and that
will fill in the columns between the first date and the
last date. First date is 2008-8-1, got the date,
thats the index, got values for apple, ibm.

I want you to add one column there at the end and call it cash.

You'll get two columns, date and stuff.

