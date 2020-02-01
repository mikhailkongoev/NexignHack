package com.nexign.aggregator.producer.api;

import com.nexign.aggregator.model.BusinessEvent;
import lombok.NonNull;

public interface Producer {
    void produce(@NonNull BusinessEvent event);
}
