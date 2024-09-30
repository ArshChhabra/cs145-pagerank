# Pagerank Project

In this project, you will create a simple search engine for the website <https://www.lawfareblog.com>.
This website provides legal analysis on US national security issues.
You will use pagerank to return only the most important results from this website in your search engine.

**Due date:** Sunday, ~~22~~ 29 September at midnight

**Late Policy:** You lose $2^{(i-1)}$ points, where i is the number of days late.

<!--
**Computation:**
This project has low computational requirements.
You should be able to complete it on your own laptops.
-->

**Collaboration Policy:**
Do whatever will help you learn,
but be an adult.
You may talk to other students and use Google/ChatGPT.
Recall that you will have an in-person oral exam on this material and the exam is worth many more points.
The main purpose of this project is to help prepare you for the exam.

## Background

**Data:**

The `data` folder contains two files that store example "web graphs".
The file `small.csv.gz` contains the example graph from the *Deeper Inside Pagerank* paper.
This is a small graph, so we can manually inspect the contents of this file with the following command:
```
$ zcat data/small.csv.gz
source,target
1,2
1,3
3,1
3,2
3,5
4,5
4,6
5,6
5,4
6,4
```

> **Recall:**
> The `cat` terminal command outputs the contents of a file to stdout, and the `zcat` command first decompressed a gzipped file and then outputs the decompressed contents.
>
> In python, we can use the built-in `gzip` module to access gzipped files.
> The following python code is equivalent to the bash code above:
>
> ```
> >>> import gzip
> >>> fin = gzip.open('data/small.csv.gz', mode='rt')
> >>> print(fin.read())
> source,target
> 1,2
> 1,3
> 3,1
> 3,2
> 3,5
> 4,5
> 4,6
> 5,6
> 5,4
> 6,4
> ```
>
> There are many terminal commands throughout these instructions.
> If you haven't used the terminal before, and so these commands are unfamiliar, that's okay.
> I'd be happy to explain them in office hours,
> or there are many tutors in the QCL available who can help.
> (There are no tutors for this class specifically, but anyone who has taken CSCI046 or CSCI133 with me will be able to help with the terminal.)
>
> Furthermore, you don't "need" to understand the terminal commands in detail,
> since you are not required to run these commands or to create your own.
> The important part is to understand the English language description of what the commands are doing,
> and to understand that this is just how I computed what the English language text is describing.

As you can see, the graph is stored as a CSV file.
The first line is a header,
and each subsequent line stores a single edge in the graph.
The first column contains the source node of the edge and the second column the target node.
The file is assumed to be sorted alphabetically.

The second data file `lawfareblog.csv.gz` contains the link structure for the lawfare blog.
Let's take a look at the first 10 of these lines:
```
$ zcat data/lawfareblog.csv.gz | head
source,target
www.lawfareblog.com/,www.lawfareblog.com/topic/interrogation
www.lawfareblog.com/,www.lawfareblog.com/upcoming-events
www.lawfareblog.com/,www.lawfareblog.com/
www.lawfareblog.com/,www.lawfareblog.com/our-comments-policy
www.lawfareblog.com/,www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
www.lawfareblog.com/,www.lawfareblog.com/topic/lawfare-research-paper-series
www.lawfareblog.com/,www.lawfareblog.com/topic/book-reviews
www.lawfareblog.com/,www.lawfareblog.com/documents-related-mueller-investigation
www.lawfareblog.com/,www.lawfareblog.com/topic/international-law-loac
```
You can see that in this file, the node names are URLs.
Semantically, each line corresponds to an HTML `<a>` tag that is contained in the source webpage and links to the target webpage.

We can use the following command to count the total number of links in the file:
```
$ zcat data/lawfareblog.csv.gz | wc -l
1610789
```
Since every link corresponds to a non-zero entry in the $P$ matrix,
this is also the value of $\text{nnz}(P)$.
(Technically, we should subtract 1 from this value since the `wc -l` command also counts the header line, not just the data lines.)

To get the dimensions of $P$, we need to count the total number of nodes in the graph.
The following command achieves this by: decompressing the file, extracting the first column, removing all duplicate lines, then counting the results.
```
$ zcat data/lawfareblog.csv.gz | cut -f1 -d, | uniq | wc -l
25761
```
This matrix is large enough that computing matrix products for dense matrices takes several minutes on a single CPU.
Fortunately, however, the matrix is very sparse.
The following python code computes the fraction of entries in the matrix with non-zero values:
```
>>> 1610788 / (25760**2)
0.0024274297384360172
```
Thus, by using sparse matrix operations, we will be able to speed up the code significantly.

**Code:**

The `pagerank.py` file contains code for loading the graph CSV files and searching through their nodes for key phrases.
For example, you can perform a search for all nodes (i.e. urls) that mention the string `corona` with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --search_query=corona
```

> **NOTE:**
> It will take about 10 seconds to load and parse the data files.
> All the other computation happens essentially instantly.

Currently, the pagerank of the nodes is not currently being calculated correctly, and so the webpages are returned in an arbitrary order.
Your task in this assignment will be to fix these calculations in order to have the most important results (i.e. highest pagerank results) returned first.

## Task 1: the power method

Implement the `WebGraph.power_method` function in `pagerank.py` for computing the pagerank vector by fixing the [`FIXME: Task 1` annotation](https://github.com/mikeizbicki/cmc-csci145-math166/blob/81ed5d2b75f5bc23b8de93805c29321ab431ed9b/topic01_computation_pagerank/project/pagerank.py#L144).

> **NOTE:**
> The power method is the only data mining algorithm you will implement in class.
> You are implementing it because there are no standard library implementations available.
> Why?
> 1. The runtime is heavily dependent on the data structures used to store the graph data.
>    Different applications will need to use different data structures.
> 1. It is "trivial" to implement.
>    My solution to this homework is <10 lines of code.

**Part 1:**

To check that your implementation is working,
you should run the program on the `data/small.csv.gz` graph.
For my implementation, I get the following output.
```
$ python3 pagerank.py --data=data/small.csv.gz --verbose
DEBUG:root:computing indices
DEBUG:root:computing values
DEBUG:root:i=0 residual=2.5629e-01
DEBUG:root:i=1 residual=1.1841e-01
DEBUG:root:i=2 residual=7.0701e-02
DEBUG:root:i=3 residual=3.1815e-02
DEBUG:root:i=4 residual=2.0497e-02
DEBUG:root:i=5 residual=1.0108e-02
DEBUG:root:i=6 residual=6.3716e-03
DEBUG:root:i=7 residual=3.4228e-03
DEBUG:root:i=8 residual=2.0879e-03
DEBUG:root:i=9 residual=1.1750e-03
DEBUG:root:i=10 residual=7.0131e-04
DEBUG:root:i=11 residual=4.0321e-04
DEBUG:root:i=12 residual=2.3800e-04
DEBUG:root:i=13 residual=1.3812e-04
DEBUG:root:i=14 residual=8.1083e-05
DEBUG:root:i=15 residual=4.7251e-05
DEBUG:root:i=16 residual=2.7704e-05
DEBUG:root:i=17 residual=1.6164e-05
DEBUG:root:i=18 residual=9.4778e-06
DEBUG:root:i=19 residual=5.5066e-06
DEBUG:root:i=20 residual=3.2042e-06
DEBUG:root:i=21 residual=1.8612e-06
DEBUG:root:i=22 residual=1.1283e-06
DEBUG:root:i=23 residual=6.1907e-07
INFO:root:rank=0 pagerank=6.6270e-01 url=4
INFO:root:rank=1 pagerank=5.2179e-01 url=6
INFO:root:rank=2 pagerank=4.1434e-01 url=5
INFO:root:rank=3 pagerank=2.3175e-01 url=2
INFO:root:rank=4 pagerank=1.8590e-01 url=3
INFO:root:rank=5 pagerank=1.6917e-01 url=1
```
Yours likely won't be identical (due to minor implementation details and weird floating point issues), but it should be similar.
In particular, the ranking of the nodes/urls should be the same order.

> **NOTE:**
> The `--verbose` flag causes all of the lines beginning with `DEBUG` to be printed.
> By default, only lines beginning with `INFO` are printed.

> **NOTE:**
> There are no automated test cases to pass for this assignment.
> Test cases for algorithms involving floating point computations are hard to write and understand.
> Minor-seeming implementations details can have large impacts on the final result.
> These software engineering issues are beyond the scope of this class.
>
> Instructions for how I will grade your homework are contained in the [submission section](#submission) at the end of this document.

**Part 2:**

The `pagerank.py` file has an option `--search_query`, which takes a string as a parameter.
If this argument is used, then the program returns all nodes that match the query string sorted according to their pagerank.
Essentially, this gives us the most important pages related to our query.

Again, you may not get the exact same results as me,
but you should get similar results to the examples I've shown below.
Verify that you do in fact get similar results.

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=8.9224e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=7.0390e-04 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=6.9153e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=6.7041e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=6.6256e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=6.5046e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=7 pagerank=6.3620e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=8 pagerank=6.1248e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
INFO:root:rank=9 pagerank=6.0187e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
INFO:root:rank=0 pagerank=5.7826e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=5.2338e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
INFO:root:rank=2 pagerank=5.1297e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
INFO:root:rank=3 pagerank=4.6599e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
INFO:root:rank=4 pagerank=4.5934e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=5 pagerank=4.3071e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=6 pagerank=4.0935e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
INFO:root:rank=7 pagerank=3.7591e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
INFO:root:rank=8 pagerank=3.4509e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
INFO:root:rank=9 pagerank=3.4484e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
INFO:root:rank=0 pagerank=4.5746e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
INFO:root:rank=1 pagerank=4.4174e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
INFO:root:rank=2 pagerank=2.6928e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=3 pagerank=1.9391e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
INFO:root:rank=4 pagerank=1.5452e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
INFO:root:rank=5 pagerank=1.5357e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
INFO:root:rank=6 pagerank=1.5258e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
INFO:root:rank=7 pagerank=1.4221e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=9 pagerank=1.1463e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force
```

**Part 3:**

The webgraph of lawfareblog.com (i.e. the $P$ matrix) naturally contains a lot of structure.
For example, essentially all pages on the domain have links to the root page <https://lawfareblog.com/> and other "non-article" pages like <https://www.lawfareblog.com/topics> and <https://www.lawfareblog.com/subscribe-lawfare>.
These pages therefore have a large pagerank.
We can get a list of the pages with the largest pagerank by running

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz
INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/topics
INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
```

Most of these pages are not very interesting, however, because they are not articles,
and usually when we are performing a web search, we only want articles.

This raises the question: How can we find the most important articles filtering out the non-article pages?
The answer is to modify the $P$ matrix by removing all links to non-article pages.

This raises another question: How do we know if a link is a non-article page?
Unfortunately, this is a hard question to answer with 100% accuracy,
but there are many methods that get us most of the way there.
One easy to implement method is to compute what's called the "in-link ratio" of each node (i.e. the total number of edges with the node as a target divided by the total number of nodes),
and then remove nodes from the search results with too-high of a ratio.
The intuition is that non-article pages often appear in the menu of a webpage, and so have links from almost all of the other webpages;
but article-webpages are unlikely to appear on a menu and so will only have a small number of links from other webpages.
The `--filter_ratio` parameter causes the code to remove all pages that have an in-link ratio larger than the provided value.

Using this option, we can estimate the most important articles on the domain with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
```
Notice that the urls in this list look much more like articles than the urls in the previous list.

When Google calculates their $P$ matrix for the web,
they use a similar (but much more complicated) process to modify the $P$ matrix in order to reduce spam results.
The exact formula they use is a jealously guarded secret that they update continuously.

In the case above, notice that we have accidentally removed the blog's most popular article (<https://www.lawfareblog.com/snowden-revelations>).
The blog editors believed that Snowden's revelations about NSA spying are so important that they directly put a link to the article on the menu.
So every single webpage in the domain links to the Snowden article,
and our "anti-spam" `--filter-ratio` argument removed this article from the list.
In general, it is a challenging open problem to remove spam from pagerank results,
and all current solutions rely on careful human tuning and still have lots of false positives and false negatives.

**Part 4:**

Recall from the reading that the runtime of pagerank depends heavily on the eigengap of the $\bar{\bar P}$ matrix,
and that this eigengap is bounded by the alpha parameter.

Run the following four commands:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
```
You should notice that the last command takes considerably more iterations to compute the pagerank vector.
(My code takes 685 iterations for this call, and about 10 iterations for all the others.)

This raises the question: Why does the second command (with the `--alpha` option but without the `--filter_ratio`) option not take a long time to run?
The answer is that the $P$ graph for <https://www.lawfareblog.com> naturally has a large eigengap and so is fast to compute for all alpha values,
but the modified graph does not have a large eigengap and so requires a small alpha for fast convergence.

Changing the value of alpha also gives us very different pagerank rankings.
For example, 
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --alpha=0.99999
INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.0552e-01 url=www.lawfareblog.com/cost-using-zero-days
INFO:root:rank=3 pagerank=3.1755e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
INFO:root:rank=4 pagerank=2.2040e-02 url=www.lawfareblog.com/events
INFO:root:rank=5 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
INFO:root:rank=6 pagerank=1.6026e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
INFO:root:rank=7 pagerank=1.6023e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
INFO:root:rank=8 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
INFO:root:rank=9 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
```

Which of these rankings is better is entirely subjective,
and the only way to know if you have the "best" alpha for your application is to try several variations and see what is best.

> **NOTE:**
> It should be "obvious" to you that large alpha values imply that the structure of the webgraph has more influence on the final result,
> and small alpha values ignore the structure of the webgraph.
> Recall that the word "obvious" means that it follows directly from the definition,
> but you may still need to sit and meditate on the definition for a long period of time.

If large alphas are good for your application, you can see that there is a trade-off between quality answers and algorithmic runtime.
We'll be exploring this trade-off more formally in class over the rest of the semester.

## Task 2: the personalization vector

The most interesting applications of pagerank involve the personalization vector.
Implement the `WebGraph.make_personalization_vector` function so that it outputs a personalization vector tuned for the input query.
The pseudocode for the function is:
```
for each index in the personalization vector:
    get the url for the index (see the _index_to_url function)
    check if the url satisfies the input query (see the url_satisfies_query function)
    if so, set the corresponding index to one
normalize the vector
```

**Part 1:**

The command line argument `--personalization_vector_query` will use the function you created above to augment your search with a custom personalization vector.
If you've implemented the function correctly,
you should get results similar to:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
```

Notice that these results are significantly different than when using the `--search_query` option:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --search_query='corona'
INFO:root:rank=0 pagerank=8.1320e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=1 pagerank=7.7908e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=2 pagerank=5.2262e-03 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
INFO:root:rank=3 pagerank=3.9584e-03 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=4 pagerank=3.8114e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=5 pagerank=3.3973e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=6 pagerank=3.3633e-03 url=www.lawfareblog.com/cyberlaw-podcast-how-israel-fighting-coronavirus
INFO:root:rank=7 pagerank=3.3557e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=8 pagerank=3.2160e-03 url=www.lawfareblog.com/congress-needs-coronavirus-failsafe-its-too-late
INFO:root:rank=9 pagerank=3.1036e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
```

Which results are better?
Again, that depends on what you mean by "better."
With the `--personalization_vector_query` option,
a webpage is important only if other coronavirus webpages also think it's important;
with the `--search_query` option,
a webpage is important if any other webpage thinks it's important.
You'll notice that in the later example, many of the webpages are about Congressional proceedings related to the coronavirus.
From a strictly coronavirus perspective, these are not very important webpages.
But in the broader context of national security, these are very important webpages.

Google engineers spend TONs of time fine-tuning their pagerank personalization vectors to remove spam webpages.
Exactly how they do this is another one of their secrets that they don't publicly talk about.

**Part 2:**

Another use of the `--personalization_vector_query` option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus.
This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their `corona` importance,
but removes webpages mentioning `corona` from the results.
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america
```
You can see that there are many urls about concepts that are obviously related like "covid", "clinical trials", and "quarantine",
but this algorithm also finds articles about Chinese propaganda and Trump's policy decisions.
Both of these articles are highly relevant to coronavirus discussions,
but a simple keyword search for corona or related terms would not find these articles.
The vast majority of industry data mining work is finding clever uses of standard algorithms.

<!--
**Part 3:**

Select another topic related to national security.
You should experiment with a national security topic other than the coronavirus.
For example, find out what articles are important to the `iran` topic but do not contain the word `iran`.
Your goal should be to discover what topics that www.lawfareblog.com considers to be related to the national security topic you choose.
-->

## Submission

1. Create a new repo on github (not a fork of this repo).
    Ensure that all of the project files are copied from this folder into your new repo.

1. As you complete the tasks above:
    Run the corresponding commands below, and paste their output into the code blocks.
    Please ensure correct markdown formatting.
   
   Task 1, part 1:
   ```
   $ python3 pagerank.py --data=data/small.csv.gz --verbose
   
    Arshs-MacBook-Pro-3:PageRankProject arshchhabra$ python3 pagerank.py --data=data/small.csv.gz --verbose
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=0.2562914192676544
    DEBUG:root:i=1 residual=0.11841227114200592
    DEBUG:root:i=2 residual=0.07070134580135345
    DEBUG:root:i=3 residual=0.03181542828679085
    DEBUG:root:i=4 residual=0.020496590062975883
    DEBUG:root:i=5 residual=0.01010835450142622
    DEBUG:root:i=6 residual=0.006371544674038887
    DEBUG:root:i=7 residual=0.0034227892756462097
    DEBUG:root:i=8 residual=0.002087961183860898
    DEBUG:root:i=9 residual=0.0011749734403565526
    DEBUG:root:i=10 residual=0.0007012754795141518
    DEBUG:root:i=11 residual=0.00040320929838344455
    DEBUG:root:i=12 residual=0.00023798426263965666
    DEBUG:root:i=13 residual=0.00013812065299134701
    DEBUG:root:i=14 residual=8.108324982458726e-05
    DEBUG:root:i=15 residual=4.7269360948121175e-05
    DEBUG:root:i=16 residual=2.7704918466042727e-05
    DEBUG:root:i=17 residual=1.6170568414963782e-05
    DEBUG:root:i=18 residual=9.479118489252869e-06
    DEBUG:root:i=19 residual=5.4782999541203026e-06
    DEBUG:root:i=20 residual=3.2123323308042018e-06
    DEBUG:root:i=21 residual=1.8802053318722756e-06
    DEBUG:root:i=22 residual=1.1228398761886638e-06
    DEBUG:root:i=23 residual=6.322027275018627e-07
    INFO:root:rank=0 pagerank=6.6270e-01 url=4
    INFO:root:rank=1 pagerank=5.2179e-01 url=6
    INFO:root:rank=2 pagerank=4.1434e-01 url=5
    INFO:root:rank=3 pagerank=2.3175e-01 url=2
    INFO:root:rank=4 pagerank=1.8590e-01 url=3
    INFO:root:rank=5 pagerank=1.6917e-01 url=1
   ```

   Task 1, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'

    Arshs-MacBook-Pro-3:PageRankProject arshchhabra$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
    INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
    INFO:root:rank=1 pagerank=8.9228e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
    INFO:root:rank=2 pagerank=7.0394e-04 url=www.lawfareblog.com/britains-coronavirus-response
    INFO:root:rank=3 pagerank=6.9157e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
    INFO:root:rank=4 pagerank=6.7045e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
    INFO:root:rank=5 pagerank=6.6260e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
    INFO:root:rank=6 pagerank=6.5050e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
    INFO:root:rank=7 pagerank=6.3623e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
    INFO:root:rank=8 pagerank=6.1252e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
    INFO:root:rank=9 pagerank=6.0191e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'

    Arshs-MacBook-Pro-3:PageRankProject arshchhabra$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
    INFO:root:rank=0 pagerank=5.7827e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=5.2340e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
    INFO:root:rank=2 pagerank=5.1298e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
    INFO:root:rank=3 pagerank=4.6601e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
    INFO:root:rank=4 pagerank=4.5935e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
    INFO:root:rank=5 pagerank=4.3073e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
    INFO:root:rank=6 pagerank=4.0936e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
    INFO:root:rank=7 pagerank=3.7592e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
    INFO:root:rank=8 pagerank=3.4510e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
    INFO:root:rank=9 pagerank=3.4486e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors
       

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'

    Arshs-MacBook-Pro-3:PageRankProject arshchhabra$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
    INFO:root:rank=0 pagerank=4.5748e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
    INFO:root:rank=1 pagerank=4.4176e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
    INFO:root:rank=2 pagerank=2.6929e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
    INFO:root:rank=3 pagerank=1.9392e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
    INFO:root:rank=4 pagerank=1.5453e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
    INFO:root:rank=5 pagerank=1.5358e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
    INFO:root:rank=6 pagerank=1.5258e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
    INFO:root:rank=7 pagerank=1.4222e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
    INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
    INFO:root:rank=9 pagerank=1.1463e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force
       
   ```

   Task 1, part 3:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz

    Arshs-MacBook-Pro-3:PageRankProject arshchhabra$ python3 pagerank.py --data=data/lawfareblog.csv.gz
    INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
    INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
    INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/topics

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2

    Arshs-MacBook-Pro-3:PageRankProject arshchhabra$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
    INFO:root:rank=0 pagerank=3.4697e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=2.9522e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
    INFO:root:rank=4 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
    INFO:root:rank=5 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
    INFO:root:rank=6 pagerank=1.5072e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
    INFO:root:rank=7 pagerank=1.4958e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
    INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
    INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
   ```

   Task 1, part 4:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose
   
    Arshs-MacBook-Pro-3:PageRankProject arshchhabra$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=1.3793749809265137
    DEBUG:root:i=1 residual=0.11642683297395706
    DEBUG:root:i=2 residual=0.07496178895235062
    DEBUG:root:i=3 residual=0.031702104955911636
    DEBUG:root:i=4 residual=0.0174466110765934
    DEBUG:root:i=5 residual=0.008526231162250042
    DEBUG:root:i=6 residual=0.00444182800129056
    DEBUG:root:i=7 residual=0.0022433267440646887
    DEBUG:root:i=8 residual=0.001149666146375239
    DEBUG:root:i=9 residual=0.0005811726441606879
    DEBUG:root:i=10 residual=0.00029266104684211314
    DEBUG:root:i=11 residual=0.00014553753135260195
    DEBUG:root:i=12 residual=7.151532918214798e-05
    DEBUG:root:i=13 residual=3.476878555375151e-05
    DEBUG:root:i=14 residual=1.5952729881973937e-05
    DEBUG:root:i=15 residual=6.454707545344718e-06
    DEBUG:root:i=16 residual=2.470087110850727e-06
    DEBUG:root:i=17 residual=8.236708595177333e-07
    INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
    INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
    INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/topics

   
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999

    Arshs-MacBook-Pro-3:PageRankProject arshchhabra$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=1.384641170501709
    DEBUG:root:i=1 residual=0.07088145613670349
    DEBUG:root:i=2 residual=0.01882273517549038
    DEBUG:root:i=3 residual=0.006958308629691601
    DEBUG:root:i=4 residual=0.0027358275838196278
    DEBUG:root:i=5 residual=0.0010345610789954662
    DEBUG:root:i=6 residual=0.0003774643409997225
    DEBUG:root:i=7 residual=0.0001353343395749107
    DEBUG:root:i=8 residual=4.822430855710991e-05
    DEBUG:root:i=9 residual=1.717166742309928e-05
    DEBUG:root:i=10 residual=6.1166115301602986e-06
    DEBUG:root:i=11 residual=2.1727698822360253e-06
    DEBUG:root:i=12 residual=7.75930004692782e-07
    INFO:root:rank=0 pagerank=2.8859e-01 url=www.lawfareblog.com/snowden-revelations
    INFO:root:rank=1 pagerank=2.8859e-01 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=2.8859e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=3 pagerank=2.8859e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=2.8859e-01 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=2.8859e-01 url=www.lawfareblog.com/topics
    INFO:root:rank=6 pagerank=2.8859e-01 url=www.lawfareblog.com/masthead
    INFO:root:rank=7 pagerank=2.8859e-01 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=2.8859e-01 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=2.8859e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    

   
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2

    Arshs-MacBook-Pro-3:PageRankProject arshchhabra$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=1.2609769105911255
    DEBUG:root:i=1 residual=0.4985710680484772
    DEBUG:root:i=2 residual=0.13418613374233246
    DEBUG:root:i=3 residual=0.0692228302359581
    DEBUG:root:i=4 residual=0.023409796878695488
    DEBUG:root:i=5 residual=0.010187179781496525
    DEBUG:root:i=6 residual=0.00490697892382741
    DEBUG:root:i=7 residual=0.002280232962220907
    DEBUG:root:i=8 residual=0.0010745070176199079
    DEBUG:root:i=9 residual=0.0005251269903965294
    DEBUG:root:i=10 residual=0.00026976881781592965
    DEBUG:root:i=11 residual=0.00014569450286217034
    DEBUG:root:i=12 residual=8.226578938774765e-05
    DEBUG:root:i=13 residual=4.813347550225444e-05
    DEBUG:root:i=14 residual=2.8801283406210132e-05
    DEBUG:root:i=15 residual=1.7420436051907018e-05
    DEBUG:root:i=16 residual=1.0539955837884918e-05
    DEBUG:root:i=17 residual=6.396006028808188e-06
    DEBUG:root:i=18 residual=3.848239430226386e-06
    DEBUG:root:i=19 residual=2.298697609148803e-06
    DEBUG:root:i=20 residual=1.3677324659511214e-06
    DEBUG:root:i=21 residual=8.154062811627227e-07
    INFO:root:rank=0 pagerank=3.4697e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=2.9522e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
    INFO:root:rank=4 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
    INFO:root:rank=5 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
    INFO:root:rank=6 pagerank=1.5072e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
    INFO:root:rank=7 pagerank=1.4958e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
    INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
    INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull



   
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999

    Arshs-MacBook-Pro-3:PageRankProject arshchhabra$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=1.2827692031860352
    DEBUG:root:i=1 residual=0.5695649981498718
    DEBUG:root:i=2 residual=0.382994681596756
    DEBUG:root:i=3 residual=0.21739362180233002
    DEBUG:root:i=4 residual=0.140450581908226
    DEBUG:root:i=5 residual=0.1085134968161583
    DEBUG:root:i=6 residual=0.09284142404794693
    DEBUG:root:i=7 residual=0.08225550502538681
    DEBUG:root:i=8 residual=0.07338894158601761
    DEBUG:root:i=9 residual=0.06561233103275299
    DEBUG:root:i=10 residual=0.0590965561568737
    DEBUG:root:i=11 residual=0.05417540296912193
    DEBUG:root:i=12 residual=0.0511169508099556
    DEBUG:root:i=13 residual=0.04999380558729172
    DEBUG:root:i=14 residual=0.05060896649956703
    DEBUG:root:i=15 residual=0.05252620205283165
    DEBUG:root:i=16 residual=0.05518876016139984
    DEBUG:root:i=17 residual=0.05803852155804634
    DEBUG:root:i=18 residual=0.06059242784976959
    DEBUG:root:i=19 residual=0.06247842311859131
    DEBUG:root:i=20 residual=0.06345325708389282
    DEBUG:root:i=21 residual=0.06340522319078445
    DEBUG:root:i=22 residual=0.06234562769532204
    DEBUG:root:i=23 residual=0.06038373336195946
    DEBUG:root:i=24 residual=0.057693902403116226
    DEBUG:root:i=25 residual=0.05447976663708687
    DEBUG:root:i=26 residual=0.05094278231263161
    DEBUG:root:i=27 residual=0.04726114124059677
    DEBUG:root:i=28 residual=0.043578632175922394
    DEBUG:root:i=29 residual=0.04000166058540344
    DEBUG:root:i=30 residual=0.03660226985812187
    DEBUG:root:i=31 residual=0.03342437744140625
    DEBUG:root:i=32 residual=0.030489342287182808
    DEBUG:root:i=33 residual=0.027803203091025352
    DEBUG:root:i=34 residual=0.025360682979226112
    DEBUG:root:i=35 residual=0.023149972781538963
    DEBUG:root:i=36 residual=0.021155361086130142
    DEBUG:root:i=37 residual=0.01935921236872673
    DEBUG:root:i=38 residual=0.017743559554219246
    DEBUG:root:i=39 residual=0.016290821135044098
    DEBUG:root:i=40 residual=0.014984318986535072
    DEBUG:root:i=41 residual=0.013808634132146835
    DEBUG:root:i=42 residual=0.0127497473731637
    DEBUG:root:i=43 residual=0.011794931255280972
    DEBUG:root:i=44 residual=0.010932877659797668
    DEBUG:root:i=45 residual=0.010153351351618767
    DEBUG:root:i=46 residual=0.009447420947253704
    DEBUG:root:i=47 residual=0.008807037025690079
    DEBUG:root:i=48 residual=0.00822513084858656
    DEBUG:root:i=49 residual=0.007695464882999659
    DEBUG:root:i=50 residual=0.007212572265416384
    DEBUG:root:i=51 residual=0.006771522108465433
    DEBUG:root:i=52 residual=0.006367973051965237
    DEBUG:root:i=53 residual=0.005998097360134125
    DEBUG:root:i=54 residual=0.0056585767306387424
    DEBUG:root:i=55 residual=0.0053463466465473175
    DEBUG:root:i=56 residual=0.0050587342120707035
    DEBUG:root:i=57 residual=0.004793429747223854
    DEBUG:root:i=58 residual=0.004548235796391964
    DEBUG:root:i=59 residual=0.004321314860135317
    DEBUG:root:i=60 residual=0.004110970068722963
    DEBUG:root:i=61 residual=0.00391573878005147
    DEBUG:root:i=62 residual=0.003734209341928363
    DEBUG:root:i=63 residual=0.003565173363313079
    DEBUG:root:i=64 residual=0.0034076254814863205
    DEBUG:root:i=65 residual=0.003260540310293436
    DEBUG:root:i=66 residual=0.0031229890882968903
    DEBUG:root:i=67 residual=0.002994265640154481
    DEBUG:root:i=68 residual=0.0028736209496855736
    DEBUG:root:i=69 residual=0.002760381205007434
    DEBUG:root:i=70 residual=0.0026539776008576155
    DEBUG:root:i=71 residual=0.0025539167691022158
    DEBUG:root:i=72 residual=0.002459658542647958
    DEBUG:root:i=73 residual=0.002370771486312151
    DEBUG:root:i=74 residual=0.002286874456331134
    DEBUG:root:i=75 residual=0.0022076282184571028
    DEBUG:root:i=76 residual=0.002132637891918421
    DEBUG:root:i=77 residual=0.002061636419966817
    DEBUG:root:i=78 residual=0.0019943402148783207
    DEBUG:root:i=79 residual=0.0019305136520415545
    DEBUG:root:i=80 residual=0.0018698670901358128
    DEBUG:root:i=81 residual=0.0018122527981176972
    DEBUG:root:i=82 residual=0.0017574202502146363
    DEBUG:root:i=83 residual=0.0017052055336534977
    DEBUG:root:i=84 residual=0.0016554546309635043
    DEBUG:root:i=85 residual=0.0016080221394076943
    DEBUG:root:i=86 residual=0.0015627118991687894
    DEBUG:root:i=87 residual=0.0015194345032796264
    DEBUG:root:i=88 residual=0.0014780519995838404
    DEBUG:root:i=89 residual=0.001438467181287706
    DEBUG:root:i=90 residual=0.0014005504781380296
    DEBUG:root:i=91 residual=0.0013642284320667386
    DEBUG:root:i=92 residual=0.0013293903321027756
    DEBUG:root:i=93 residual=0.0012959633022546768
    DEBUG:root:i=94 residual=0.0012638646876439452
    DEBUG:root:i=95 residual=0.0012330401223152876
    DEBUG:root:i=96 residual=0.0012033784296363592
    DEBUG:root:i=97 residual=0.0011748677352443337
    DEBUG:root:i=98 residual=0.0011474041966721416
    DEBUG:root:i=99 residual=0.0011209577787667513
    DEBUG:root:i=100 residual=0.001095480751246214
    DEBUG:root:i=101 residual=0.0010709036141633987
    DEBUG:root:i=102 residual=0.0010471886489540339
    DEBUG:root:i=103 residual=0.0010242939461022615
    DEBUG:root:i=104 residual=0.0010021800408139825
    DEBUG:root:i=105 residual=0.0009808088652789593
    DEBUG:root:i=106 residual=0.0009601510828360915
    DEBUG:root:i=107 residual=0.0009401601273566484
    DEBUG:root:i=108 residual=0.0009208220872096717
    DEBUG:root:i=109 residual=0.0009020877187140286
    DEBUG:root:i=110 residual=0.0008839471847750247
    DEBUG:root:i=111 residual=0.0008663663174957037
    DEBUG:root:i=112 residual=0.0008493210771121085
    DEBUG:root:i=113 residual=0.0008327716495841742
    DEBUG:root:i=114 residual=0.0008167123887687922
    DEBUG:root:i=115 residual=0.0008011438185349107
    DEBUG:root:i=116 residual=0.0007860065088607371
    DEBUG:root:i=117 residual=0.0007713016238994896
    DEBUG:root:i=118 residual=0.0007570052985101938
    DEBUG:root:i=119 residual=0.0007431135163642466
    DEBUG:root:i=120 residual=0.0007295955438166857
    DEBUG:root:i=121 residual=0.0007164462585933506
    DEBUG:root:i=122 residual=0.0007036398164927959
    DEBUG:root:i=123 residual=0.0006911731325089931
    DEBUG:root:i=124 residual=0.0006790345651097596
    DEBUG:root:i=125 residual=0.0006672001327387989
    DEBUG:root:i=126 residual=0.0006556716398335993
    DEBUG:root:i=127 residual=0.0006444274331443012
    DEBUG:root:i=128 residual=0.0006334587233141065
    DEBUG:root:i=129 residual=0.0006227657431736588
    DEBUG:root:i=130 residual=0.000612325151450932
    DEBUG:root:i=131 residual=0.000602136948145926
    DEBUG:root:i=132 residual=0.0005921903066337109
    DEBUG:root:i=133 residual=0.0005824631080031395
    DEBUG:root:i=134 residual=0.000572981487493962
    DEBUG:root:i=135 residual=0.0005637037102133036
    DEBUG:root:i=136 residual=0.0005546361790038645
    DEBUG:root:i=137 residual=0.0005457779625430703
    DEBUG:root:i=138 residual=0.0005371182924136519
    DEBUG:root:i=139 residual=0.0005286375526338816
    DEBUG:root:i=140 residual=0.0005203467444516718
    DEBUG:root:i=141 residual=0.0005122313159517944
    DEBUG:root:i=142 residual=0.0005042926059104502
    DEBUG:root:i=143 residual=0.0004965200205333531
    DEBUG:root:i=144 residual=0.000488906807731837
    DEBUG:root:i=145 residual=0.00048145439359359443
    DEBUG:root:i=146 residual=0.0004741552984341979
    DEBUG:root:i=147 residual=0.00046699648373760283
    DEBUG:root:i=148 residual=0.0004599886015057564
    DEBUG:root:i=149 residual=0.00045311806024983525
    DEBUG:root:i=150 residual=0.0004463914083316922
    DEBUG:root:i=151 residual=0.00043978425674140453
    DEBUG:root:i=152 residual=0.0004333136312197894
    DEBUG:root:i=153 residual=0.000426965270889923
    DEBUG:root:i=154 residual=0.00042074115481227636
    DEBUG:root:i=155 residual=0.0004146253631915897
    DEBUG:root:i=156 residual=0.00040863692993298173
    DEBUG:root:i=157 residual=0.00040275498759001493
    DEBUG:root:i=158 residual=0.00039698072941973805
    DEBUG:root:i=159 residual=0.00039131444646045566
    DEBUG:root:i=160 residual=0.00038575136568397284
    DEBUG:root:i=161 residual=0.00038029084680601954
    DEBUG:root:i=162 residual=0.0003749237221200019
    DEBUG:root:i=163 residual=0.00036965476465411484
    DEBUG:root:i=164 residual=0.000364480831194669
    DEBUG:root:i=165 residual=0.00035939394729211926
    DEBUG:root:i=166 residual=0.0003543994389474392
    DEBUG:root:i=167 residual=0.00034949113614857197
    DEBUG:root:i=168 residual=0.0003446676128078252
    DEBUG:root:i=169 residual=0.00033992473618127406
    DEBUG:root:i=170 residual=0.00033526530023664236
    DEBUG:root:i=171 residual=0.0003306881699245423
    DEBUG:root:i=172 residual=0.00032618356635794044
    DEBUG:root:i=173 residual=0.00032175713567994535
    DEBUG:root:i=174 residual=0.0003173986915498972
    DEBUG:root:i=175 residual=0.00031312124338001013
    DEBUG:root:i=176 residual=0.00030890764901414514
    DEBUG:root:i=177 residual=0.00030476777465082705
    DEBUG:root:i=178 residual=0.00030069032800383866
    DEBUG:root:i=179 residual=0.0002966813917737454
    DEBUG:root:i=180 residual=0.0002927372697740793
    DEBUG:root:i=181 residual=0.00028885522624477744
    DEBUG:root:i=182 residual=0.00028503683279268444
    DEBUG:root:i=183 residual=0.0002812762395478785
    DEBUG:root:i=184 residual=0.0002775773173198104
    DEBUG:root:i=185 residual=0.00027393485652282834
    DEBUG:root:i=186 residual=0.0002703513309825212
    DEBUG:root:i=187 residual=0.0002668215602170676
    DEBUG:root:i=188 residual=0.00026334929862059653
    DEBUG:root:i=189 residual=0.0002599266008473933
    DEBUG:root:i=190 residual=0.000256557745160535
    DEBUG:root:i=191 residual=0.00025324514717794955
    DEBUG:root:i=192 residual=0.0002499764086678624
    DEBUG:root:i=193 residual=0.0002467571175657213
    DEBUG:root:i=194 residual=0.0002435886417515576
    DEBUG:root:i=195 residual=0.00024046775070019066
    DEBUG:root:i=196 residual=0.00023739466269034892
    DEBUG:root:i=197 residual=0.00023436496849171817
    DEBUG:root:i=198 residual=0.00023138168035075068
    DEBUG:root:i=199 residual=0.0002284416841575876
    DEBUG:root:i=200 residual=0.00022554700262844563
    DEBUG:root:i=201 residual=0.00022269370674621314
    DEBUG:root:i=202 residual=0.00021988025400787592
    DEBUG:root:i=203 residual=0.00021710633882321417
    DEBUG:root:i=204 residual=0.00021437476971186697
    DEBUG:root:i=205 residual=0.00021168545936234295
    DEBUG:root:i=206 residual=0.00020903089898638427
    DEBUG:root:i=207 residual=0.0002064148138742894
    DEBUG:root:i=208 residual=0.00020383905211929232
    DEBUG:root:i=209 residual=0.00020129616314079612
    DEBUG:root:i=210 residual=0.00019879329192917794
    DEBUG:root:i=211 residual=0.00019632511248346418
    DEBUG:root:i=212 residual=0.0001938904169946909
    DEBUG:root:i=213 residual=0.00019148667342960835
    DEBUG:root:i=214 residual=0.000189119455171749
    DEBUG:root:i=215 residual=0.0001867868850240484
    DEBUG:root:i=216 residual=0.0001844867510953918
    DEBUG:root:i=217 residual=0.00018221852951683104
    DEBUG:root:i=218 residual=0.00017998000839725137
    DEBUG:root:i=219 residual=0.00017777190078049898
    DEBUG:root:i=220 residual=0.00017559689877089113
    DEBUG:root:i=221 residual=0.00017345126252621412
    DEBUG:root:i=222 residual=0.00017133417713921517
    DEBUG:root:i=223 residual=0.00016924398369155824
    DEBUG:root:i=224 residual=0.00016718590632081032
    DEBUG:root:i=225 residual=0.00016515386232640594
    DEBUG:root:i=226 residual=0.0001631476916372776
    DEBUG:root:i=227 residual=0.00016117132327053696
    DEBUG:root:i=228 residual=0.00015922088641673326
    DEBUG:root:i=229 residual=0.00015729661390651017
    DEBUG:root:i=230 residual=0.0001553973270347342
    DEBUG:root:i=231 residual=0.00015352203627116978
    DEBUG:root:i=232 residual=0.00015167474339250475
    DEBUG:root:i=233 residual=0.00014985182497184724
    DEBUG:root:i=234 residual=0.00014805021055508405
    DEBUG:root:i=235 residual=0.00014627441123593599
    DEBUG:root:i=236 residual=0.0001445223460905254
    DEBUG:root:i=237 residual=0.00014279405877459794
    DEBUG:root:i=238 residual=0.00014108563482295722
    DEBUG:root:i=239 residual=0.00013940165808890015
    DEBUG:root:i=240 residual=0.00013773910177405924
    DEBUG:root:i=241 residual=0.00013609851885121316
    DEBUG:root:i=242 residual=0.00013447992387227714
    DEBUG:root:i=243 residual=0.00013288174523040652
    DEBUG:root:i=244 residual=0.00013130342995282263
    DEBUG:root:i=245 residual=0.00012974477431271225
    DEBUG:root:i=246 residual=0.00012820886331610382
    DEBUG:root:i=247 residual=0.000126693383208476
    DEBUG:root:i=248 residual=0.00012519500160124153
    DEBUG:root:i=249 residual=0.00012371732736937702
    DEBUG:root:i=250 residual=0.00012225670798216015
    DEBUG:root:i=251 residual=0.00012081705062882975
    DEBUG:root:i=252 residual=0.00011939539399463683
    DEBUG:root:i=253 residual=0.00011799062485806644
    DEBUG:root:i=254 residual=0.00011660345626296476
    DEBUG:root:i=255 residual=0.000115233997348696
    DEBUG:root:i=256 residual=0.00011388392158551142
    DEBUG:root:i=257 residual=0.0001125498311012052
    DEBUG:root:i=258 residual=0.00011123240983579308
    DEBUG:root:i=259 residual=0.0001099312721635215
    DEBUG:root:i=260 residual=0.00010864646174013615
    DEBUG:root:i=261 residual=0.0001073772000381723
    DEBUG:root:i=262 residual=0.00010612420737743378
    DEBUG:root:i=263 residual=0.00010488546831766143
    DEBUG:root:i=264 residual=0.00010366570495534688
    DEBUG:root:i=265 residual=0.00010246012971037999
    DEBUG:root:i=266 residual=0.00010126592678716406
    DEBUG:root:i=267 residual=0.00010009075049310923
    DEBUG:root:i=268 residual=9.892740490613505e-05
    DEBUG:root:i=269 residual=9.778103412827477e-05
    DEBUG:root:i=270 residual=9.664873505244032e-05
    DEBUG:root:i=271 residual=9.552918345434591e-05
    DEBUG:root:i=272 residual=9.442529699299484e-05
    DEBUG:root:i=273 residual=9.333375783171505e-05
    DEBUG:root:i=274 residual=9.225479880115017e-05
    DEBUG:root:i=275 residual=9.119007154367864e-05
    DEBUG:root:i=276 residual=9.013712406158447e-05
    DEBUG:root:i=277 residual=8.909984171623364e-05
    DEBUG:root:i=278 residual=8.807306585367769e-05
    DEBUG:root:i=279 residual=8.705855725565925e-05
    DEBUG:root:i=280 residual=8.605829498264939e-05
    DEBUG:root:i=281 residual=8.506923040840775e-05
    DEBUG:root:i=282 residual=8.40914435684681e-05
    DEBUG:root:i=283 residual=8.312641875818372e-05
    DEBUG:root:i=284 residual=8.21727589936927e-05
    DEBUG:root:i=285 residual=8.123197039822116e-05
    DEBUG:root:i=286 residual=8.029958553379402e-05
    DEBUG:root:i=287 residual=7.938141789054498e-05
    DEBUG:root:i=288 residual=7.847448432585225e-05
    DEBUG:root:i=289 residual=7.75759108364582e-05
    DEBUG:root:i=290 residual=7.669039769098163e-05
    DEBUG:root:i=291 residual=7.581372483400628e-05
    DEBUG:root:i=292 residual=7.495026511605829e-05
    DEBUG:root:i=293 residual=7.409446698147804e-05
    DEBUG:root:i=294 residual=7.325045589823276e-05
    DEBUG:root:i=295 residual=7.241472485475242e-05
    DEBUG:root:i=296 residual=7.159051165217534e-05
    DEBUG:root:i=297 residual=7.077628106344491e-05
    DEBUG:root:i=298 residual=6.99723168509081e-05
    DEBUG:root:i=299 residual=6.917717109899968e-05
    DEBUG:root:i=300 residual=6.839146954007447e-05
    DEBUG:root:i=301 residual=6.761537224519998e-05
    DEBUG:root:i=302 residual=6.684708205284551e-05
    DEBUG:root:i=303 residual=6.60881632938981e-05
    DEBUG:root:i=304 residual=6.534046406159177e-05
    DEBUG:root:i=305 residual=6.459974247263744e-05
    DEBUG:root:i=306 residual=6.386781751643866e-05
    DEBUG:root:i=307 residual=6.31450311630033e-05
    DEBUG:root:i=308 residual=6.243189272936434e-05
    DEBUG:root:i=309 residual=6.172443681862205e-05
    DEBUG:root:i=310 residual=6.1028596974210814e-05
    DEBUG:root:i=311 residual=6.033860699972138e-05
    DEBUG:root:i=312 residual=5.965885065961629e-05
    DEBUG:root:i=313 residual=5.898505332879722e-05
    DEBUG:root:i=314 residual=5.831904854858294e-05
    DEBUG:root:i=315 residual=5.766136018792167e-05
    DEBUG:root:i=316 residual=5.700870315195061e-05
    DEBUG:root:i=317 residual=5.6368615332758054e-05
    DEBUG:root:i=318 residual=5.573332600761205e-05
    DEBUG:root:i=319 residual=5.510561459232122e-05
    DEBUG:root:i=320 residual=5.448622687254101e-05
    DEBUG:root:i=321 residual=5.387388591771014e-05
    DEBUG:root:i=322 residual=5.3267554903868586e-05
    DEBUG:root:i=323 residual=5.2670311561087146e-05
    DEBUG:root:i=324 residual=5.207786671235226e-05
    DEBUG:root:i=325 residual=5.14936436957214e-05
    DEBUG:root:i=326 residual=5.091718048788607e-05
    DEBUG:root:i=327 residual=5.034566856920719e-05
    DEBUG:root:i=328 residual=4.978112337994389e-05
    DEBUG:root:i=329 residual=4.922209700453095e-05
    DEBUG:root:i=330 residual=4.866944073000923e-05
    DEBUG:root:i=331 residual=4.81243223475758e-05
    DEBUG:root:i=332 residual=4.7586978325853124e-05
    DEBUG:root:i=333 residual=4.705556420958601e-05
    DEBUG:root:i=334 residual=4.65277953480836e-05
    DEBUG:root:i=335 residual=4.600682950695045e-05
    DEBUG:root:i=336 residual=4.5492368371924385e-05
    DEBUG:root:i=337 residual=4.4985092245042324e-05
    DEBUG:root:i=338 residual=4.448345862329006e-05
    DEBUG:root:i=339 residual=4.3985357478959486e-05
    DEBUG:root:i=340 residual=4.349433947936632e-05
    DEBUG:root:i=341 residual=4.3009593355236575e-05
    DEBUG:root:i=342 residual=4.2530948121566325e-05
    DEBUG:root:i=343 residual=4.205440927762538e-05
    DEBUG:root:i=344 residual=4.1586768929846585e-05
    DEBUG:root:i=345 residual=4.112552051083185e-05
    DEBUG:root:i=346 residual=4.0665181586518884e-05
    DEBUG:root:i=347 residual=4.021518907393329e-05
    DEBUG:root:i=348 residual=3.976502193836495e-05
    DEBUG:root:i=349 residual=3.932314211851917e-05
    DEBUG:root:i=350 residual=3.8886824768269435e-05
    DEBUG:root:i=351 residual=3.845303217531182e-05
    DEBUG:root:i=352 residual=3.8026628317311406e-05
    DEBUG:root:i=353 residual=3.7603920645779e-05
    DEBUG:root:i=354 residual=3.718604421010241e-05
    DEBUG:root:i=355 residual=3.6770637962035835e-05
    DEBUG:root:i=356 residual=3.6363693652674556e-05
    DEBUG:root:i=357 residual=3.596194801502861e-05
    DEBUG:root:i=358 residual=3.5562119592214e-05
    DEBUG:root:i=359 residual=3.5167096939403564e-05
    DEBUG:root:i=360 residual=3.477724749245681e-05
    DEBUG:root:i=361 residual=3.4391949156997725e-05
    DEBUG:root:i=362 residual=3.4011336538242176e-05
    DEBUG:root:i=363 residual=3.363309951964766e-05
    DEBUG:root:i=364 residual=3.326152727822773e-05
    DEBUG:root:i=365 residual=3.2892909075599164e-05
    DEBUG:root:i=366 residual=3.252940223319456e-05
    DEBUG:root:i=367 residual=3.216862751287408e-05
    DEBUG:root:i=368 residual=3.181332431267947e-05
    DEBUG:root:i=369 residual=3.14601456921082e-05
    DEBUG:root:i=370 residual=3.111304249614477e-05
    DEBUG:root:i=371 residual=3.076938810409047e-05
    DEBUG:root:i=372 residual=3.0429280741373077e-05
    DEBUG:root:i=373 residual=3.009207830473315e-05
    DEBUG:root:i=374 residual=2.9759572498733178e-05
    DEBUG:root:i=375 residual=2.9431355869746767e-05
    DEBUG:root:i=376 residual=2.910491093643941e-05
    DEBUG:root:i=377 residual=2.8784166715922765e-05
    DEBUG:root:i=378 residual=2.8468863092712127e-05
    DEBUG:root:i=379 residual=2.815286279655993e-05
    DEBUG:root:i=380 residual=2.784239040920511e-05
    DEBUG:root:i=381 residual=2.753437547653448e-05
    DEBUG:root:i=382 residual=2.7231397325522266e-05
    DEBUG:root:i=383 residual=2.693104943318758e-05
    DEBUG:root:i=384 residual=2.6633859306457452e-05
    DEBUG:root:i=385 residual=2.634010888868943e-05
    DEBUG:root:i=386 residual=2.6050574888358824e-05
    DEBUG:root:i=387 residual=2.5763352823560126e-05
    DEBUG:root:i=388 residual=2.547871736169327e-05
    DEBUG:root:i=389 residual=2.5198802177328616e-05
    DEBUG:root:i=390 residual=2.4921233489294536e-05
    DEBUG:root:i=391 residual=2.4648270482430235e-05
    DEBUG:root:i=392 residual=2.4375349312322214e-05
    DEBUG:root:i=393 residual=2.4107139324769378e-05
    DEBUG:root:i=394 residual=2.384330946370028e-05
    DEBUG:root:i=395 residual=2.35805655393051e-05
    DEBUG:root:i=396 residual=2.3322267225012183e-05
    DEBUG:root:i=397 residual=2.306366150151007e-05
    DEBUG:root:i=398 residual=2.2811838789493777e-05
    DEBUG:root:i=399 residual=2.2560487195733003e-05
    DEBUG:root:i=400 residual=2.2313246518024243e-05
    DEBUG:root:i=401 residual=2.206761018896941e-05
    DEBUG:root:i=402 residual=2.1824167561135255e-05
    DEBUG:root:i=403 residual=2.1584006390185095e-05
    DEBUG:root:i=404 residual=2.1347546862671152e-05
    DEBUG:root:i=405 residual=2.111408866767306e-05
    DEBUG:root:i=406 residual=2.088092514895834e-05
    DEBUG:root:i=407 residual=2.0652118109865114e-05
    DEBUG:root:i=408 residual=2.042625601461623e-05
    DEBUG:root:i=409 residual=2.020216197706759e-05
    DEBUG:root:i=410 residual=1.998043808271177e-05
    DEBUG:root:i=411 residual=1.9761671865126118e-05
    DEBUG:root:i=412 residual=1.9544517272152007e-05
    DEBUG:root:i=413 residual=1.9330096620251425e-05
    DEBUG:root:i=414 residual=1.9118797354167327e-05
    DEBUG:root:i=415 residual=1.8909071513917297e-05
    DEBUG:root:i=416 residual=1.8700638975133188e-05
    DEBUG:root:i=417 residual=1.8495535186957568e-05
    DEBUG:root:i=418 residual=1.8294949768460356e-05
    DEBUG:root:i=419 residual=1.809379318729043e-05
    DEBUG:root:i=420 residual=1.7895079508889467e-05
    DEBUG:root:i=421 residual=1.7699256204650737e-05
    DEBUG:root:i=422 residual=1.7504647985333577e-05
    DEBUG:root:i=423 residual=1.7312142517766915e-05
    DEBUG:root:i=424 residual=1.7124988517025486e-05
    DEBUG:root:i=425 residual=1.6936137399170548e-05
    DEBUG:root:i=426 residual=1.67511279869359e-05
    DEBUG:root:i=427 residual=1.656908352742903e-05
    DEBUG:root:i=428 residual=1.6387881260016002e-05
    DEBUG:root:i=429 residual=1.6207823136937805e-05
    DEBUG:root:i=430 residual=1.603011151019018e-05
    DEBUG:root:i=431 residual=1.5855475794523954e-05
    DEBUG:root:i=432 residual=1.568203151691705e-05
    DEBUG:root:i=433 residual=1.551070818095468e-05
    DEBUG:root:i=434 residual=1.5341202015406452e-05
    DEBUG:root:i=435 residual=1.5173180145211518e-05
    DEBUG:root:i=436 residual=1.5006762623670511e-05
    DEBUG:root:i=437 residual=1.4844340512354393e-05
    DEBUG:root:i=438 residual=1.4681081665912643e-05
    DEBUG:root:i=439 residual=1.4520000149786938e-05
    DEBUG:root:i=440 residual=1.4362086403707508e-05
    DEBUG:root:i=441 residual=1.4206082596501801e-05
    DEBUG:root:i=442 residual=1.4048372577235568e-05
    DEBUG:root:i=443 residual=1.3894882613385562e-05
    DEBUG:root:i=444 residual=1.3744010175287258e-05
    DEBUG:root:i=445 residual=1.3591955394076649e-05
    DEBUG:root:i=446 residual=1.3445658623822965e-05
    DEBUG:root:i=447 residual=1.3298604244482704e-05
    DEBUG:root:i=448 residual=1.3154184671293478e-05
    DEBUG:root:i=449 residual=1.3009597751079127e-05
    DEBUG:root:i=450 residual=1.2867512850789353e-05
    DEBUG:root:i=451 residual=1.2727473404083867e-05
    DEBUG:root:i=452 residual=1.2588195204443764e-05
    DEBUG:root:i=453 residual=1.2449861969798803e-05
    DEBUG:root:i=454 residual=1.2313998013269156e-05
    DEBUG:root:i=455 residual=1.218174566020025e-05
    DEBUG:root:i=456 residual=1.2048421922372654e-05
    DEBUG:root:i=457 residual=1.1916034964087885e-05
    DEBUG:root:i=458 residual=1.1785077731474303e-05
    DEBUG:root:i=459 residual=1.1655813977995422e-05
    DEBUG:root:i=460 residual=1.1530388292158023e-05
    DEBUG:root:i=461 residual=1.1403903044993058e-05
    DEBUG:root:i=462 residual=1.1279687896603718e-05
    DEBUG:root:i=463 residual=1.1157932021887973e-05
    DEBUG:root:i=464 residual=1.1035695933969691e-05
    DEBUG:root:i=465 residual=1.091505509975832e-05
    DEBUG:root:i=466 residual=1.0796024071169086e-05
    DEBUG:root:i=467 residual=1.0678735634428449e-05
    DEBUG:root:i=468 residual=1.0562293027760461e-05
    DEBUG:root:i=469 residual=1.044701821228955e-05
    DEBUG:root:i=470 residual=1.0333371392334811e-05
    DEBUG:root:i=471 residual=1.022010746964952e-05
    DEBUG:root:i=472 residual=1.0109613867825828e-05
    DEBUG:root:i=473 residual=9.998437235481106e-06
    DEBUG:root:i=474 residual=9.890016372082755e-06
    DEBUG:root:i=475 residual=9.781902917893603e-06
    DEBUG:root:i=476 residual=9.674667126091663e-06
    DEBUG:root:i=477 residual=9.570003385306336e-06
    DEBUG:root:i=478 residual=9.46587897487916e-06
    DEBUG:root:i=479 residual=9.363450772070792e-06
    DEBUG:root:i=480 residual=9.261035302188247e-06
    DEBUG:root:i=481 residual=9.161781235889066e-06
    DEBUG:root:i=482 residual=9.060815500561148e-06
    DEBUG:root:i=483 residual=8.96311212272849e-06
    DEBUG:root:i=484 residual=8.864221854310017e-06
    DEBUG:root:i=485 residual=8.769107807893306e-06
    DEBUG:root:i=486 residual=8.673815500515047e-06
    DEBUG:root:i=487 residual=8.578875167586375e-06
    DEBUG:root:i=488 residual=8.485170837957412e-06
    DEBUG:root:i=489 residual=8.393668395001441e-06
    DEBUG:root:i=490 residual=8.302582500618882e-06
    DEBUG:root:i=491 residual=8.211287422454916e-06
    DEBUG:root:i=492 residual=8.122734470816795e-06
    DEBUG:root:i=493 residual=8.034672646317631e-06
    DEBUG:root:i=494 residual=7.946052392071579e-06
    DEBUG:root:i=495 residual=7.86102009442402e-06
    DEBUG:root:i=496 residual=7.774804544169456e-06
    DEBUG:root:i=497 residual=7.689991434745025e-06
    DEBUG:root:i=498 residual=7.606912276969524e-06
    DEBUG:root:i=499 residual=7.524584816565039e-06
    DEBUG:root:i=500 residual=7.441346497216728e-06
    DEBUG:root:i=501 residual=7.361483767454047e-06
    DEBUG:root:i=502 residual=7.281781563506229e-06
    DEBUG:root:i=503 residual=7.20217622074415e-06
    DEBUG:root:i=504 residual=7.125036063371226e-06
    DEBUG:root:i=505 residual=7.045297934382688e-06
    DEBUG:root:i=506 residual=6.969634796405444e-06
    DEBUG:root:i=507 residual=6.8944646045565605e-06
    DEBUG:root:i=508 residual=6.818719612056157e-06
    DEBUG:root:i=509 residual=6.744815891579492e-06
    DEBUG:root:i=510 residual=6.6715283537632786e-06
    DEBUG:root:i=511 residual=6.598796517209848e-06
    DEBUG:root:i=512 residual=6.527247933263425e-06
    DEBUG:root:i=513 residual=6.457357812905684e-06
    DEBUG:root:i=514 residual=6.3864017647574656e-06
    DEBUG:root:i=515 residual=6.317966381175211e-06
    DEBUG:root:i=516 residual=6.249037596717244e-06
    DEBUG:root:i=517 residual=6.1810574152332265e-06
    DEBUG:root:i=518 residual=6.11300083619426e-06
    DEBUG:root:i=519 residual=6.046987891750177e-06
    DEBUG:root:i=520 residual=5.9818771660502534e-06
    DEBUG:root:i=521 residual=5.917954240430845e-06
    DEBUG:root:i=522 residual=5.8535983953333925e-06
    DEBUG:root:i=523 residual=5.789432634628611e-06
    DEBUG:root:i=524 residual=5.7265738178102765e-06
    DEBUG:root:i=525 residual=5.663855063176015e-06
    DEBUG:root:i=526 residual=5.603111276286654e-06
    DEBUG:root:i=527 residual=5.542601229535649e-06
    DEBUG:root:i=528 residual=5.483710992848501e-06
    DEBUG:root:i=529 residual=5.4225879466685e-06
    DEBUG:root:i=530 residual=5.363947366276989e-06
    DEBUG:root:i=531 residual=5.305666491040029e-06
    DEBUG:root:i=532 residual=5.246694854577072e-06
    DEBUG:root:i=533 residual=5.189935109228827e-06
    DEBUG:root:i=534 residual=5.134646016813349e-06
    DEBUG:root:i=535 residual=5.078482445242116e-06
    DEBUG:root:i=536 residual=5.02241982758278e-06
    DEBUG:root:i=537 residual=4.969529527443228e-06
    DEBUG:root:i=538 residual=4.915641511615831e-06
    DEBUG:root:i=539 residual=4.861646630160976e-06
    DEBUG:root:i=540 residual=4.808876383322058e-06
    DEBUG:root:i=541 residual=4.756809175887611e-06
    DEBUG:root:i=542 residual=4.704894763563061e-06
    DEBUG:root:i=543 residual=4.65344146505231e-06
    DEBUG:root:i=544 residual=4.602952230925439e-06
    DEBUG:root:i=545 residual=4.55456256531761e-06
    DEBUG:root:i=546 residual=4.503599029703764e-06
    DEBUG:root:i=547 residual=4.455896487343125e-06
    DEBUG:root:i=548 residual=4.4085436456953175e-06
    DEBUG:root:i=549 residual=4.360499133326812e-06
    DEBUG:root:i=550 residual=4.311758402764099e-06
    DEBUG:root:i=551 residual=4.2651413423300255e-06
    DEBUG:root:i=552 residual=4.218981302983593e-06
    DEBUG:root:i=553 residual=4.1735038394108415e-06
    DEBUG:root:i=554 residual=4.128844921069685e-06
    DEBUG:root:i=555 residual=4.084096417500405e-06
    DEBUG:root:i=556 residual=4.038702627440216e-06
    DEBUG:root:i=557 residual=3.994242433691397e-06
    DEBUG:root:i=558 residual=3.954073690692894e-06
    DEBUG:root:i=559 residual=3.908794496965129e-06
    DEBUG:root:i=560 residual=3.866799943352817e-06
    DEBUG:root:i=561 residual=3.8245766518230084e-06
    DEBUG:root:i=562 residual=3.7835231978533557e-06
    DEBUG:root:i=563 residual=3.7408265143312747e-06
    DEBUG:root:i=564 residual=3.7018407965661027e-06
    DEBUG:root:i=565 residual=3.6600956718757516e-06
    DEBUG:root:i=566 residual=3.624071723606903e-06
    DEBUG:root:i=567 residual=3.582793851819588e-06
    DEBUG:root:i=568 residual=3.5441332784102997e-06
    DEBUG:root:i=569 residual=3.5052876228292007e-06
    DEBUG:root:i=570 residual=3.4673289519560058e-06
    DEBUG:root:i=571 residual=3.429727939874283e-06
    DEBUG:root:i=572 residual=3.3930641620827373e-06
    DEBUG:root:i=573 residual=3.3562268981768284e-06
    DEBUG:root:i=574 residual=3.3207270462298766e-06
    DEBUG:root:i=575 residual=3.2832340366439894e-06
    DEBUG:root:i=576 residual=3.2487857879459625e-06
    DEBUG:root:i=577 residual=3.2130719773704186e-06
    DEBUG:root:i=578 residual=3.1773442970006727e-06
    DEBUG:root:i=579 residual=3.1440220027434407e-06
    DEBUG:root:i=580 residual=3.109185627181432e-06
    DEBUG:root:i=581 residual=3.075895619986113e-06
    DEBUG:root:i=582 residual=3.0456581043836195e-06
    DEBUG:root:i=583 residual=3.0111580144875916e-06
    DEBUG:root:i=584 residual=2.976860741910059e-06
    DEBUG:root:i=585 residual=2.9434875159495277e-06
    DEBUG:root:i=586 residual=2.912649733843864e-06
    DEBUG:root:i=587 residual=2.881689852074487e-06
    DEBUG:root:i=588 residual=2.851884573829011e-06
    DEBUG:root:i=589 residual=2.8203451165609295e-06
    DEBUG:root:i=590 residual=2.7885157578566577e-06
    DEBUG:root:i=591 residual=2.7592193418968236e-06
    DEBUG:root:i=592 residual=2.734692543526762e-06
    DEBUG:root:i=593 residual=2.700199502214673e-06
    DEBUG:root:i=594 residual=2.6706381959229475e-06
    DEBUG:root:i=595 residual=2.6398706722829957e-06
    DEBUG:root:i=596 residual=2.614489176266943e-06
    DEBUG:root:i=597 residual=2.5832016490312526e-06
    DEBUG:root:i=598 residual=2.560505208748509e-06
    DEBUG:root:i=599 residual=2.5301574169134255e-06
    DEBUG:root:i=600 residual=2.5006579562614206e-06
    DEBUG:root:i=601 residual=2.47532625508029e-06
    DEBUG:root:i=602 residual=2.4478113118675537e-06
    DEBUG:root:i=603 residual=2.4209416551457252e-06
    DEBUG:root:i=604 residual=2.395224782958394e-06
    DEBUG:root:i=605 residual=2.368646391914808e-06
    DEBUG:root:i=606 residual=2.3485592919314513e-06
    DEBUG:root:i=607 residual=2.319371105841128e-06
    DEBUG:root:i=608 residual=2.2948304376768647e-06
    DEBUG:root:i=609 residual=2.2704798539052717e-06
    DEBUG:root:i=610 residual=2.251584419354913e-06
    DEBUG:root:i=611 residual=2.2234210064198123e-06
    DEBUG:root:i=612 residual=2.1961757283861516e-06
    DEBUG:root:i=613 residual=2.1714079139201203e-06
    DEBUG:root:i=614 residual=2.147912255168194e-06
    DEBUG:root:i=615 residual=2.125621222148766e-06
    DEBUG:root:i=616 residual=2.102483449561987e-06
    DEBUG:root:i=617 residual=2.0785428205272183e-06
    DEBUG:root:i=618 residual=2.0566931198118255e-06
    DEBUG:root:i=619 residual=2.0349586975498823e-06
    DEBUG:root:i=620 residual=2.0121492525504436e-06
    DEBUG:root:i=621 residual=1.9923768377339e-06
    DEBUG:root:i=622 residual=1.969691311387578e-06
    DEBUG:root:i=623 residual=1.9463500393612776e-06
    DEBUG:root:i=624 residual=1.927253151734476e-06
    DEBUG:root:i=625 residual=1.9057463305216515e-06
    DEBUG:root:i=626 residual=1.884748030533956e-06
    DEBUG:root:i=627 residual=1.8673703152671806e-06
    DEBUG:root:i=628 residual=1.843891027419886e-06
    DEBUG:root:i=629 residual=1.8254065707878908e-06
    DEBUG:root:i=630 residual=1.8045061551674735e-06
    DEBUG:root:i=631 residual=1.7847744402388344e-06
    DEBUG:root:i=632 residual=1.766861714713741e-06
    DEBUG:root:i=633 residual=1.7461998140788637e-06
    DEBUG:root:i=634 residual=1.7289767129113898e-06
    DEBUG:root:i=635 residual=1.7128378431152669e-06
    DEBUG:root:i=636 residual=1.6912448472794495e-06
    DEBUG:root:i=637 residual=1.6720611029086285e-06
    DEBUG:root:i=638 residual=1.653996605455177e-06
    DEBUG:root:i=639 residual=1.6386048855565605e-06
    DEBUG:root:i=640 residual=1.617391717445571e-06
    DEBUG:root:i=641 residual=1.600669293111423e-06
    DEBUG:root:i=642 residual=1.587370775268937e-06
    DEBUG:root:i=643 residual=1.5686970300521352e-06
    DEBUG:root:i=644 residual=1.5500621657338343e-06
    DEBUG:root:i=645 residual=1.532781652713311e-06
    DEBUG:root:i=646 residual=1.5174008467511158e-06
    DEBUG:root:i=647 residual=1.5021388435343397e-06
    DEBUG:root:i=648 residual=1.482361994931125e-06
    DEBUG:root:i=649 residual=1.467606011829048e-06
    DEBUG:root:i=650 residual=1.4508583490169258e-06
    DEBUG:root:i=651 residual=1.4372830037245876e-06
    DEBUG:root:i=652 residual=1.420653120476345e-06
    DEBUG:root:i=653 residual=1.4066832818571129e-06
    DEBUG:root:i=654 residual=1.3897490589442896e-06
    DEBUG:root:i=655 residual=1.3756498447037302e-06
    DEBUG:root:i=656 residual=1.3622099004351185e-06
    DEBUG:root:i=657 residual=1.3446037883113604e-06
    DEBUG:root:i=658 residual=1.3307783319760347e-06
    DEBUG:root:i=659 residual=1.3165516747903894e-06
    DEBUG:root:i=660 residual=1.3048856999375857e-06
    DEBUG:root:i=661 residual=1.288504222429765e-06
    DEBUG:root:i=662 residual=1.274061332878773e-06
    DEBUG:root:i=663 residual=1.2606400332515477e-06
    DEBUG:root:i=664 residual=1.2484734952522558e-06
    DEBUG:root:i=665 residual=1.2335607380009606e-06
    DEBUG:root:i=666 residual=1.219712771671766e-06
    DEBUG:root:i=667 residual=1.206725414704124e-06
    DEBUG:root:i=668 residual=1.1934091617149534e-06
    DEBUG:root:i=669 residual=1.1815344578280929e-06
    DEBUG:root:i=670 residual=1.167345203612058e-06
    DEBUG:root:i=671 residual=1.1580368663999252e-06
    DEBUG:root:i=672 residual=1.1442798495409079e-06
    DEBUG:root:i=673 residual=1.131452108893427e-06
    DEBUG:root:i=674 residual=1.1185842367922305e-06
    DEBUG:root:i=675 residual=1.1070404752899776e-06
    DEBUG:root:i=676 residual=1.0923960189757054e-06
    DEBUG:root:i=677 residual=1.096402002076502e-06
    DEBUG:root:i=678 residual=1.0742927543105907e-06
    DEBUG:root:i=679 residual=1.060759245774534e-06
    DEBUG:root:i=680 residual=1.0497816447241348e-06
    DEBUG:root:i=681 residual=1.038765049088397e-06
    DEBUG:root:i=682 residual=1.0268878440911067e-06
    DEBUG:root:i=683 residual=1.021697585201764e-06
    DEBUG:root:i=684 residual=1.0035560080723371e-06
    DEBUG:root:i=685 residual=9.938793255059863e-07
    INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.0552e-01 url=www.lawfareblog.com/cost-using-zero-days
    INFO:root:rank=3 pagerank=3.1757e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
    INFO:root:rank=4 pagerank=2.2040e-02 url=www.lawfareblog.com/events
    INFO:root:rank=5 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
    INFO:root:rank=6 pagerank=1.6026e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
    INFO:root:rank=7 pagerank=1.6023e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
    INFO:root:rank=8 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
    INFO:root:rank=9 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
   
   ```

   Task 2, part 1:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'

    Arshs-MacBook-Pro-3:PageRankProject arshchhabra$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
    INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
    INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
    INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
    INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
    INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
    INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
       
   ```

   Task 2, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'


    Arshs-MacBook-Pro-3:PageRankProject arshchhabra$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
    INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
    INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
    INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
    INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
    INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
    INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-committee-holds-hearing-priorities-missile-defense
   ```

1. Ensure that all your changes to the `pagerank.py` and `README.md` files are committed to your repo and pushed to github.

1. Get at least 5 stars on your repo.
   (You may trade stars with other students in the class.)

   > **NOTE:**
   > 
   > Recruiters use github profiles to determine who to hire,
   > and pagerank is used to rank user profiles and projects.
   > Links in this graph correspond to who has starred/followed who's repo.
   > By getting more stars on your repo, you'll be increasing your github pagerank, which increases the likelihood that recruiters will hire you.
   > To see an example, [perform a search for `data mining`](https://github.com/search?q=data+mining).
   > Notice that the results are returned "approximately" ranked by the number of stars,
   > but because "some stars count more than others" the results are not exactly ranked by the number of stars.
   > (I asked you not to fork this repo because forks are ranked lower than non-forks.)
   >
   > In some sense, we are doing a "dual problem" to data mining by getting these stars.
   > Recruiters are using data mining to find out who the best people to recruit are,
   > and we are hacking their data mining algorithms by making those algorithms select you instead of someone else.
   >
   > If you're interested in exploring this idea further, here's a python tutorial for extracting GitHub's social graph: <https://www.oreilly.com/library/view/mining-the-social/9781449368180/ch07.html> ; if you're interested in learning more about how recruiters use github profiles, read this Hacker News post: <https://news.ycombinator.com/item?id=19413348>.

1. Submit the url of your repo to sakai.

   The assignment is worth 8 points.
   1. There are 6 parts to the output above.  (4 in Task1 and 2 in Task2.)
   1. Each part that you get incorrect will result in -2 points.  (But you cannot go negative.)
   1. Another way of phrasing this is that the first 2 parts you complete are not worth any points,
      but each part after that is worth 2 points.
