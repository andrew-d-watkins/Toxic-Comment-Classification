from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
plt.figure(figsize=(20,20))

#Clean comments
#Get all the clean text
clean_text = train_df[train_df.non_toxic == True].comment_text.values
wc= WordCloud(background_color="black", max_words=4000, stopwords=stopwords)
wc.generate(" ".join(clean_text))
plt.subplot(321)
plt.axis("off")
plt.title("Clean Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=244), alpha=0.98)

#Obscene comments
#Get all the obscene text
obscene_text = train_df[train_df.obscene == True].comment_text.values
wc= WordCloud(background_color="black", max_words=4000, stopwords=stopwords)
wc.generate(" ".join(obscene_text))
plt.subplot(322)
plt.axis("off")
plt.title("Obscene Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'Reds' , random_state=244), alpha=0.98)

#Insult comments
#Get all the insult text
insult_text = train_df[train_df.insult == True].comment_text.values
wc= WordCloud(background_color="black", max_words=4000, stopwords=stopwords)
wc.generate(" ".join(insult_text))
plt.subplot(323)
plt.axis("off")
plt.title("Insulting Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'summer' , random_state=2534), alpha=0.98)

#Severe Toxic comments
#Get all the severe toxic text
severe_toxic_text = train_df[train_df.severe_toxic == True].comment_text.values
wc= WordCloud(background_color="black", max_words=4000, stopwords=stopwords)
wc.generate(" ".join(severe_toxic_text))
plt.subplot(324)
plt.axis("off")
plt.title("Severe Toxic Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'Paired_r' , random_state=244), alpha=0.98)

#Identity hate comments
#Get all the identity hate text
identity_hate_text = train_df[train_df.identity_hate == True].comment_text.values
wc= WordCloud(background_color="black", max_words=4000, stopwords=stopwords)
wc.generate(" ".join(identity_hate_text))
plt.subplot(325)
plt.axis("off")
plt.title("Identity hate Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'plasma' , random_state=244), alpha=0.98)

#Threat comments
#Get all the threat text
threat_text = train_df[train_df.threat == True].comment_text.values
wc= WordCloud(background_color="black", max_words=4000, stopwords=stopwords)
wc.generate(" ".join(threat_text))
plt.subplot(326)
plt.axis("off")
plt.title("Threating Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'inferno' , random_state=244), alpha=0.98)

plt.show()