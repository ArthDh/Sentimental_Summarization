import torch

class MergeModel(torch.nn.Module):
    '''
            Todo
    '''
    def __init__(self, summary_model, sentiment_model):
        super().__init__()
        self.summary_model = summary_model
        self.sentiment_model = sentiment_model
    
    def forward(self, article_id, article_mask, summary_id, summary_mask):
        self.summary_out = self.summary_model(article_id,  article_mask, summary_id, summary_mask)
        self.article_sentiment = self.sentiment_model(article_id,  article_mask, labels=None)
        self.summary_sentiment = self.sentiment_model(summary_id,  summary_mask, labels=None)
        
        return [self.summary_out, self.article_sentiment, self.summary_sentiment]